"""Decide como o drone reage a um inimigo: atirar, perseguir, fugir ou
esquivar. O `RiskModel` entra aqui só para pontuar células de fuga; quem
mantém a saúde do combate (tiros, acertos, cooldown) é o próprio
`CombatController`.
"""

import time

from fuzzy import combat_aggressiveness
from world_model import BLOCKED, DANGEROUS, DIR_VECTORS, VISITED, SAFE, in_bounds

LOW_ENERGY = 40
CRITICAL_ENERGY = 25

# Alcance máximo em que vale a pena atirar quando a energia não está no
# talo. Além disso a chance de acerto cai e o tiro (-10) deixa de compensar.
MAX_ATTACK_DISTANCE = 6

# Quantos tiros errados seguidos o agente tolera antes de desistir do alvo
# e ir procurar outra coisa (o inimigo provavelmente está desviando).
FIRST_ENGAGEMENT_MISS_LIMIT = 3
MIN_SHOTS_BEFORE_HIT_RATE_CHECK = 6
MIN_ACCEPTABLE_HIT_RATE = 0.20
DISENGAGE_COOLDOWN = 8.0

# Limiares de agressividade fuzzy (energia x distância) que separam
# "atacar", "fugir" e "contra-atacar depois de levar um tiro".
FLEE_AGGRESSIVENESS_MIN = 0.30
FLEE_ON_STEPS_AGGRESSIVENESS_MIN = 0.25
COUNTERATTACK_AFTER_DAMAGE_MIN = 0.50

HUNT_TURNS_AFTER_STEPS = 2
HUNT_TURNS_AFTER_DAMAGE = 4


class CombatController:
    """Toma as decisões de combate do drone e guarda o histórico de tiros
    (acertos/erros) usado para saber quando insistir ou desistir de um alvo."""

    def __init__(self, world, risk, log=print):
        self.world = world
        self.risk = risk
        self.log = log
        self.shots_since_hit = 0
        self.shots_fired = 0
        self.shots_hit = 0
        self.awaiting_shot_result = False
        self.engage_pause_until = 0.0
        self.hunt_reason = None
        self.hunt_turns = 0

    def reset(self):
        """Zera o histórico de combate (chamado no início de cada partida)."""
        self.__init__(self.world, self.risk, self.log)

    def update_after_observation(self, observation):
        """Confere se o último tiro acertou assim que a observação chega."""
        if self.awaiting_shot_result:
            self.shots_fired += 1
            if observation.hit:
                self.shots_hit += 1
                self.shots_since_hit = 0
                self.log("[COMBATE] Hit confirmado")
            else:
                self.shots_since_hit += 1
            self.awaiting_shot_result = False
        elif observation.hit:
            self.shots_hit += 1
            self.shots_since_hit = 0

    def aggressiveness(self, energy, observation):
        """Agressividade fuzzy (energia x distância do inimigo). Sem
        distância visível, assume perto se levou tiro/ouviu passos, e longe
        caso contrário."""
        dist = observation.enemy_distance
        if dist is None:
            dist = 2 if (observation.damage or observation.steps) else 10
        return combat_aggressiveness(energy, dist)

    def line_of_fire_clear(self, x, y, direction, enemy_dist):
        """True se não há parede conhecida entre o drone e o inimigo."""
        if enemy_dist is None:
            return False
        vx, vy = DIR_VECTORS.get(direction, (0, 0))
        for step in range(1, enemy_dist + 1):
            tx, ty = x + vx * step, y + vy * step
            if not in_bounds(tx, ty):
                return False
            if self.world.grid[tx][ty] == BLOCKED:
                return False
        return True

    def miss_limit(self, enemy_dist):
        """Quantos tiros sem acerto o agente ainda tolera, dado o alcance
        atual (alvo mais perto = mais tolerância, o acerto é mais provável)."""
        if enemy_dist is None:
            return 2
        if enemy_dist <= 3:
            return 6
        if enemy_dist <= 6:
            return 4
        return 2

    def should_attack(self, energy, observation):
        """Decide se vale atirar agora. Tiro custa -10 e matar rende +1000,
        mas só compensa insistir enquanto a taxa de acerto real (aprendida
        pelos tiros já dados) sustentar essa conta."""
        enemy_dist = observation.enemy_distance
        if enemy_dist is None:
            return False
        if time.time() < self.engage_pause_until:
            return False
        if not self.line_of_fire_clear(*(self.world.position or (0, 0)),
                                       self.world.orientation, enemy_dist):
            return False
        if self.shots_hit == 0 and self.shots_fired >= FIRST_ENGAGEMENT_MISS_LIMIT:
            self.engage_pause_until = time.time() + DISENGAGE_COOLDOWN
            return False
        if self.shots_since_hit >= self.miss_limit(enemy_dist):
            return False
        if energy <= CRITICAL_ENERGY and enemy_dist > 2:
            return False
        if enemy_dist > MAX_ATTACK_DISTANCE and energy < 85:
            return False

        hit_rate = self.shots_hit / self.shots_fired if self.shots_fired else 0.5
        if self.shots_fired >= MIN_SHOTS_BEFORE_HIT_RATE_CHECK and hit_rate < MIN_ACCEPTABLE_HIT_RATE:
            return enemy_dist <= 3

        if observation.hit:
            return energy > CRITICAL_ENERGY and enemy_dist <= MAX_ATTACK_DISTANCE
        aggr = self.aggressiveness(energy, observation)
        return aggr >= 0.42 or (energy >= 80 and enemy_dist <= 7)

    def decide(self, energy, observation):
        """Estado de combate para este tick, ou None se nada exige reação
        (o chamador segue com exploração/farm normalmente)."""
        enemy_dist = observation.enemy_distance
        aggr = self.aggressiveness(energy, observation)

        if observation.damage:
            if energy <= LOW_ENERGY or aggr < COUNTERATTACK_AFTER_DAMAGE_MIN:
                return "EVADE"
            # Energia e agressividade dão pra contra-atacar em vez de só
            # fugir: gira à procura de quem atirou (ver do_hunt).
            self.hunt_reason = "damage"
            return "HUNT"

        if enemy_dist is not None:
            if self.should_attack(energy, observation):
                return "ATTACK"
            if enemy_dist <= 3 or aggr < FLEE_AGGRESSIVENESS_MIN or energy <= LOW_ENERGY:
                return "EVADE"
            if enemy_dist <= MAX_ATTACK_DISTANCE + 3 and energy > LOW_ENERGY:
                return "CHASE"

        if observation.steps and aggr < FLEE_ON_STEPS_AGGRESSIVENESS_MIN:
            return "FLEE"
        return None

    def record_shot(self):
        self.awaiting_shot_result = True

    def after_miss_limit(self, enemy_dist):
        """Se o limite de erros foi atingido, desengaja por um tempo (o
        alvo provavelmente está desviando) e avisa o chamador."""
        if self.shots_since_hit >= self.miss_limit(enemy_dist):
            self.engage_pause_until = time.time() + DISENGAGE_COOLDOWN
            self.shots_since_hit = 0
            return True
        return False

    def can_chase_forward(self, x, y, direction):
        """True se a célula à frente é segura para avançar durante um CHASE."""
        vx, vy = DIR_VECTORS.get(direction, (0, 0))
        nx, ny = x + vx, y + vy
        if not in_bounds(nx, ny):
            return False
        cell = self.world.grid[nx][ny]
        if cell == BLOCKED or cell in DANGEROUS:
            return False
        if self.world.pit_risk(nx, ny) > 1:
            return False
        return self.world.is_walkable(nx, ny, allow_unknown=True)

    def best_evade_cell(self, x, y, direction, observation):
        """Escolhe a célula vizinha mais segura para sair da linha de tiro:
        prioriza sair do eixo do ataque, células já visitadas e com mais
        saídas, penalizando zonas de risco conhecidas."""
        candidates = []
        for dir_name, (vx, vy) in DIR_VECTORS.items():
            nx, ny = x + vx, y + vy
            if not in_bounds(nx, ny):
                continue
            cell = self.world.grid[nx][ny]
            if cell == BLOCKED or cell in DANGEROUS:
                continue
            if cell not in (SAFE, VISITED):
                continue
            same_axis = (
                (direction in ("north", "south") and dir_name in ("north", "south")) or
                (direction in ("east", "west") and dir_name in ("east", "west"))
            )
            score = 0.0
            if observation.damage or observation.enemy:
                score += 10.0 if not same_axis else -2.0
            if cell == VISITED:
                score += 3.0
            score += self.world.safe_exit_count(nx, ny)
            score -= self.risk.cell_penalty((nx, ny)) * 0.08
            candidates.append((score, (nx, ny)))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def hunt_turn_budget(self):
        """Quantos giros o HUNT faz antes de desistir: uma varredura curta
        quando só ouvimos passos (alvo incerto), mais longa quando levamos
        um tiro de verdade (o atirador está em linha reta conosco)."""
        return HUNT_TURNS_AFTER_STEPS if self.hunt_reason == "steps" else HUNT_TURNS_AFTER_DAMAGE
