"""
Inteligencia do drone: Maquina de Estados Finitos + inferencia logica
(world_model) + busca A* + logica fuzzy para decisoes de combate.

Estados:
  EXPLORE  - explora a fronteira do mapa conhecido em busca de itens;
  GRAB     - pega item na celula atual (luz detectada);
  ATTACK   - inimigo na mira e agressividade fuzzy alta -> atira;
  CHASE    - inimigo na mira mas longe -> avanca para fechar distancia;
  EVADE    - levou dano/ameaca perto -> sai da linha de tiro;
  HUNT     - levou dano (atirador em linha reta) -> gira procurando-o;
  SURVEY   - varredura curta enquanto defende ponto de farm;
  RECHARGE - energia baixa -> vai ate um powerup conhecido;
  FLEE     - agressividade fuzzy baixa (fraco/ameacado) -> foge.

A transicao ATTACK/EVADE/HUNT/FLEE usa a 'agressividade' calculada por um
controlador fuzzy (fuzzy.py) sobre energia e distancia do inimigo.
"""

import os
import time

from world_model import (WorldModel, DIR_VECTORS, DANGEROUS, DANGER_PIT,
                         DANGER_BOTH, DANGER_FLASH, VISITED, BLOCKED,
                         in_bounds, neighbors)
from fuzzy import combat_aggressiveness

LOW_ENERGY = 40  # abaixo disso, busca powerup conhecido
CRITICAL_ENERGY = 25
COUNTERFIRE_AGGR = 0.65
LONG_SHOT_ENERGY = 75
POWERUP_PICKUP_ENERGY = 55  # powerup nao pontua; pegar alto demais desperdiça acao
MAX_FRONTIER_CANDIDATES = 80
QUIET_FARM_AFTER = 6.0      # sem ameaca recente, prioriza farm/camping
STEP_HUNT_TURNS = 4         # steps sem enemy: uma varredura completa basta
STEP_IGNORE_AFTER_SCAN = 5.0
DEFEND_TREASURE_MAX_WAIT = 24.0
FARM_SURVEY_INTERVAL = 6.0
FARM_SURVEY_TURNS = 4
CHASE_LOST_HUNT_TURNS = 4

TURN_LEFT_OF = {"north": "west", "west": "south", "south": "east", "east": "north"}
TURN_RIGHT_OF = {"north": "east", "east": "south", "south": "west", "west": "north"}


class DroneAgent:
    def __init__(self, game_ai, log=print):
        self.ai = game_ai
        self.world = WorldModel()
        self.log = log
        self.strategy = os.environ.get("DRONE_STRATEGY", "farm").lower()
        self.state = "EXPLORE"
        self.path = []            # caminho atual (lista de celulas)
        self.path_allows_flash = False
        self.goal = None
        self.hunt_turns = 0       # giros restantes no estado HUNT
        self.last_taken = {}      # (x,y) -> timestamp da ultima coleta
        self.tick_count = 0
        # deteccao de travamento (forward sem efeito e sem 'blocked')
        self.last_pos = None
        self.last_action = None
        self.forward_fails = 0
        self.sync_fails = 0
        # economia de combate: desengaja apos varios tiros sem acerto
        self.shots_since_hit = 0
        self.engage_pause_until = 0.0
        self.last_threat_at = time.time()
        self.ignore_steps_until = 0.0
        self.current_score = 0
        self.last_survey_at = 0.0
        self.survey_turns = 0
        # economia de acoes: instante em que ficamos sem alvos
        self.idle_since = None
        # farming: estimativa adaptativa do tempo de respawn dos itens e
        # guarda anti-spam de 'pegar' por celula
        self.respawn_est = 12.0
        self.last_grab = {}

    # ---------------- percepcao ----------------

    def sync(self):
        """Atualiza posicao/energia e observacoes do servidor
        (status + observacao pedidos em paralelo). O timeout cresce com
        falhas consecutivas: servidor lento atrasa o agente, mas nunca o
        faz agir com dados velhos."""
        timeout = 0.55 + min(self.sync_fails * 0.15, 0.80)
        return self.ai.request_sync_pair(timeout=timeout)

    def enemy_distance(self, obs):
        """Extrai enemy#dist/eneny#dist das observacoes do servidor."""
        for o in obs:
            low = o.lower()
            if low.startswith(("enemy", "eneny")):
                try:
                    return int(low.split("#")[1])
                except (IndexError, ValueError):
                    return 5
        return None

    def turn_cost(self, current_dir, target_dir):
        if current_dir == target_dir:
            return 0
        if TURN_LEFT_OF[current_dir] == target_dir or \
                TURN_RIGHT_OF[current_dir] == target_dir:
            return 1
        return 2

    def _path_action_cost(self, start, start_dir, path):
        """Custo real aproximado: passos + giros necessarios para seguir path."""
        if not path:
            return 0
        cost = 0
        current = start
        current_dir = start_dir
        dir_of = {v: k for k, v in DIR_VECTORS.items()}
        for nxt in path:
            delta = (nxt[0] - current[0], nxt[1] - current[1])
            next_dir = dir_of.get(delta)
            if next_dir is None:
                return 9999
            cost += 1 + self.turn_cost(current_dir, next_dir)
            current = nxt
            current_dir = next_dir
        return cost

    def _kill_mode(self):
        return self.strategy == "kill"

    def _killfarm_mode(self):
        return self.strategy in ("killfarm", "huntfarm", "hybrid")

    def _combat_first_mode(self):
        return self._kill_mode() or self._killfarm_mode()

    def _item_value(self, pos, energy, now):
        kind = self.world.item_spots.get(pos, "unknown")
        age = now - self.last_taken.get(pos, 0)
        maturity = max(0.0, age - self.respawn_est)
        maturity_bonus = min(250.0, maturity * 20.0)
        if kind == "treasure":
            return 1000.0 + maturity_bonus
        if kind == "unknown":
            return 650.0 + maturity_bonus
        # Powerup nao pontua diretamente; so ganha prioridade quando energia
        # baixa torna a sobrevivencia mais valiosa que economizar a acao.
        energy_need = max(0, POWERUP_PICKUP_ENERGY - energy)
        return 160.0 + energy_need * 9.0

    def _frontier_info_gain(self, pos):
        x, y = pos
        unknown_neighbors = sum(
            1 for nx, ny in neighbors(x, y)
            if self.world.grid[nx][ny] == "?"
        )
        own_bonus = 2 if self.world.grid[x][y] == "?" else 0
        return own_bonus + unknown_neighbors

    def _frontier_candidates(self, start, frontier):
        sx, sy = start

        def rank(pos):
            info = self._frontier_info_gain(pos)
            dist = abs(pos[0] - sx) + abs(pos[1] - sy)
            risk = self.world.pit_risk(*pos)
            return (-(info * 8 - dist - risk * 6), dist)

        return sorted(frontier, key=rank)[:MAX_FRONTIER_CANDIDATES]

    def _best_scored_plan(self, start, start_dir, targets, score_fn,
                          allow_flash=False):
        """Escolhe o alvo de maior valor liquido, nao apenas o mais proximo."""
        best = None
        for target in targets:
            path = self.world.a_star(start, target, allow_unknown=True,
                                     start_dir=start_dir,
                                     allow_flash=allow_flash)
            if path is None:
                goal, bfs_path = self.world.nearest_reachable(
                    start, [target], allow_flash=allow_flash)
                if goal is None:
                    continue
                path = bfs_path
            cost = self._path_action_cost(start, start_dir, path)
            if cost >= 9999:
                continue
            score = score_fn(target, path, cost)
            if best is None or score > best[0]:
                best = (score, target, path, cost)
        return best

    def _threat_seen(self, obs_l):
        return (
            "steps" in obs_l or "damage" in obs_l or "hit" in obs_l or
            any(o.startswith(("enemy", "eneny")) for o in obs_l)
        )

    def _steps_only(self, obs_l):
        return (
            "steps" in obs_l and
            "damage" not in obs_l and
            "hit" not in obs_l and
            not any(o.startswith(("enemy", "eneny")) for o in obs_l)
        )

    def _quiet_for_farm(self, now):
        return now - self.last_threat_at >= QUIET_FARM_AFTER

    def _farm_camp_targets(self):
        return [
            pos for pos, kind in self.world.item_spots.items()
            if kind in ("treasure", "unknown")
        ]

    def _camp_value(self, pos, cost, now):
        kind = self.world.item_spots.get(pos, "unknown")
        base = 1000.0 if kind == "treasure" else 650.0
        last = self.last_taken.get(pos)
        wait = 0.0 if last is None else max(0.0, self.respawn_est - (now - last))
        # Esperar parado e gratis, mas em servidor cheio esperar demais aumenta
        # chance de outro bot levar o respawn. Penaliza espera, nao proibe camping.
        return base - cost * 1.5 - wait * 6.0

    def _should_defend_current_camp(self, pos, now):
        if pos not in self._farm_camp_targets() or pos not in self.last_taken:
            return False
        wait = max(0.0, self.respawn_est - (now - self.last_taken[pos]))
        if self.current_score < 0:
            return wait <= 8.0
        return wait <= DEFEND_TREASURE_MAX_WAIT

    def _survey_due(self, now):
        return self._killfarm_mode() and \
            now - self.last_survey_at >= FARM_SURVEY_INTERVAL

    def _start_survey(self, now, reason):
        self.state = "SURVEY"
        self.survey_turns = FARM_SURVEY_TURNS
        self.last_survey_at = now
        self.log(f"[SURVEY] {reason}: varrendo arredores")

    def _front_cell(self, x, y, d):
        vx, vy = DIR_VECTORS.get(d, (0, 0))
        return x + vx, y + vy

    def _can_chase_forward(self, x, y, d):
        nx, ny = self._front_cell(x, y, d)
        if not in_bounds(nx, ny):
            return False
        cell = self.world.grid[nx][ny]
        if cell == BLOCKED or cell in DANGEROUS:
            return False
        if self.world.pit_risk(nx, ny) > 1:
            return False
        return self.world.is_walkable(nx, ny, allow_unknown=True)

    def _line_of_fire_clear(self, x, y, d, enemy_dist):
        if enemy_dist is None:
            return False
        vx, vy = DIR_VECTORS.get(d, (0, 0))
        for step in range(1, enemy_dist + 1):
            tx, ty = x + vx * step, y + vy * step
            if not in_bounds(tx, ty):
                return False
            if self.world.grid[tx][ty] == BLOCKED:
                return False
        return True

    def _miss_limit(self, enemy_dist):
        """Quanto mais longe o alvo, menor a tolerancia a tiros sem hit."""
        if self._kill_mode():
            if enemy_dist is None:
                return 4
            if enemy_dist <= 4:
                return 12
            if enemy_dist <= 8:
                return 8
            return 5
        if self._killfarm_mode():
            if enemy_dist is None:
                return 2
            if enemy_dist <= 4:
                return 10
            if enemy_dist <= 6:
                return 5
            if enemy_dist <= 8:
                return 3
            return 1
        if enemy_dist is None:
            return 2
        if enemy_dist <= 3:
            return 6
        if enemy_dist <= 6:
            return 4
        return 2

    def _should_attack(self, energy, enemy_dist, obs):
        """Decide se o tiro vale o custo (-10) neste tick."""
        obs_l = [o.lower() for o in obs]
        if enemy_dist is None:
            return False
        if self._kill_mode():
            return energy > 0 and enemy_dist <= 10
        if self._killfarm_mode():
            if "damage" in obs_l and energy <= LOW_ENERGY:
                return False
            if self.shots_since_hit >= self._miss_limit(enemy_dist):
                return False
            if "hit" in obs_l:
                return energy > LOW_ENERGY and enemy_dist <= 8
            if energy <= CRITICAL_ENERGY:
                return enemy_dist <= 2 and self.aggr >= 0.35
            if enemy_dist <= 4:
                return energy > CRITICAL_ENERGY
            if enemy_dist <= 6:
                return energy >= 45 and self.aggr >= 0.45
            if enemy_dist <= 8:
                score_pressure = self.current_score < 0
                return energy >= 80 and self.shots_since_hit == 0 and score_pressure
            return False
        if "hit" in obs_l:
            return energy > CRITICAL_ENERGY and enemy_dist <= 8
        if self.shots_since_hit >= self._miss_limit(enemy_dist):
            return False
        if energy <= CRITICAL_ENERGY:
            return enemy_dist <= 2 and self.aggr >= 0.35
        if enemy_dist <= 3:
            return self.aggr >= 0.35
        if enemy_dist <= 6:
            return self.aggr >= 0.55
        return energy >= LONG_SHOT_ENERGY and self.aggr >= 0.75

    # ---------------- atuacao de baixo nivel ----------------

    def face(self, current_dir, target_dir):
        """Executa 1 acao de giro em direcao a target_dir. Retorna True se ja alinhado."""
        if current_dir == target_dir:
            return True
        if TURN_LEFT_OF[current_dir] == target_dir:
            self.ai.send_turn_left()
            self.log(f"[ACAO] Virar a esquerda ({current_dir} -> {target_dir})")
        else:
            self.ai.send_turn_right()
            self.log(f"[ACAO] Virar a direita ({current_dir} -> {target_dir})")
        return False

    def step_towards(self, x, y, d, target):
        """Da um passo (girando se preciso) rumo a celula adjacente target."""
        dx, dy = target[0] - x, target[1] - y
        want = None
        for dir_name, (vx, vy) in DIR_VECTORS.items():
            if (vx, vy) == (dx, dy):
                want = dir_name
                break
        if want is None:
            return  # celula nao adjacente: caminho sera recalculado
        if self.face(d, want):
            self.ai.send_forward()
            self.last_action = "forward"
            self.log(f"[ACAO] Andar para frente -> {target}")
        else:
            self.last_action = "turn"

    # ---------------- decisao (FSM) ----------------

    def decide_state(self, x, y, energy, obs):
        # o servidor pode enviar "enemy#xx" ou "eneny#xx" (grafia do enunciado)
        enemy_dist = self.enemy_distance(obs)
        obs_l = [o.lower() for o in obs]
        took_damage = "damage" in obs_l
        now = time.time()
        steps_only = self._steps_only(obs_l)
        if enemy_dist is not None or took_damage or "hit" in obs_l:
            self.ignore_steps_until = 0.0
        hears_steps = "steps" in obs_l and not (
            steps_only and now < self.ignore_steps_until
        )

        # a LUZ observada e a verdade do servidor: se ha luz, ha item AGORA
        # (essencial para farming de respawns). Guarda anti-spam evita pegar
        # duas vezes antes de o servidor processar.
        if "bluelight" in obs_l:
            light = "treasure"
        elif "redlight" in obs_l:
            light = "powerup"
        elif "weaklight" in obs_l:
            light = "unknown"
        else:
            light = None
        has_item = light is not None and \
            (light != "powerup" or energy <= POWERUP_PICKUP_ENERGY) and \
            now - self.last_grab.get((x, y), 0) > 0.6
        threat_now = enemy_dist is not None or took_damage or hears_steps
        ambiguous_steps = steps_only and enemy_dist is None and not took_damage

        # agressividade fuzzy: energia x distancia do inimigo. Sem inimigo
        # visivel, dano/steps implicam inimigo perto (dist ~2); senao longe.
        if enemy_dist is not None:
            assumed_dist = enemy_dist
        elif took_damage or hears_steps:
            assumed_dist = 2
        else:
            assumed_dist = 10
        self.aggr = combat_aggressiveness(energy, assumed_dist)

        # item embaixo do drone vale sempre no modo farm; no modo killfarm so
        # quando nao ha ameaca por perto.
        if has_item and (
                not self._combat_first_mode() or
                (self._killfarm_mode() and (not threat_now or ambiguous_steps))):
            return "GRAB"
        if self._combat_first_mode() and light == "powerup" and \
                energy <= CRITICAL_ENERGY and \
                now - self.last_grab.get((x, y), 0) > 0.6:
            return "GRAB"
        if enemy_dist is not None:
            # desengajado por excesso de tiros errados? ignora e segue
            if self._combat_first_mode():
                if took_damage and energy <= LOW_ENERGY:
                    return "EVADE"
                if self._should_attack(energy, enemy_dist, obs):
                    return "ATTACK"
                if self._killfarm_mode():
                    return "EVADE" if energy <= LOW_ENERGY else "CHASE"
                return "HUNT"
            if time.time() < self.engage_pause_until and "hit" not in obs_l:
                return "EVADE" if took_damage or enemy_dist <= 2 else "EXPLORE"
            if took_damage and self.aggr < COUNTERFIRE_AGGR:
                return "EVADE"
            if self._should_attack(energy, enemy_dist, obs):
                return "ATTACK"
            if enemy_dist <= 3 or energy <= LOW_ENERGY:
                return "EVADE"
            if self.aggr < 0.25:
                return "FLEE"
            return "EXPLORE"
        if took_damage:
            # levamos tiro: o atirador esta em linha reta conosco
            return "EVADE"
        if self.state == "SURVEY" and self.survey_turns > 0:
            return "SURVEY"
        if self.state == "HUNT" and self.hunt_turns > 0:
            return "HUNT"
        if self._combat_first_mode() and hears_steps:
            return "HUNT"
        if hears_steps and self.aggr < 0.25:
            return "FLEE"
        if (not self._combat_first_mode() or self._killfarm_mode()) and \
                energy <= LOW_ENERGY and \
                self._due_spots(("powerup",), energy=0):
            return "RECHARGE"
        return "EXPLORE"

    def _due_spots(self, kinds, energy):
        """Pontos de item da memoria permanente provavelmente disponiveis:
        nunca coletados, ou coletados ha mais que a estimativa de respawn.
        Powerups so contam com energia <= 70 (senao seriam desperdicados)."""
        now = time.time()
        out = []
        for pos, kind in self.world.item_spots.items():
            if kind not in kinds:
                continue
            if kind == "powerup" and energy > POWERUP_PICKUP_ENERGY:
                continue
            if pos not in self.last_taken or \
                    now - self.last_taken[pos] >= self.respawn_est:
                out.append(pos)
        return out

    # ---------------- comportamento por estado ----------------

    def act(self):
        view = self.sync()
        if view is None:
            # resposta atrasada: descarta o tick em vez de agir com dados
            # velhos (que corromperiam o mapa e poderiam levar a um poco)
            self.sync_fails += 1
            if self.sync_fails % 20 == 1:
                self.log("[REDE] Servidor lento: aguardando dados frescos...")
            return
        self.sync_fails = 0
        x, y, d, pstate, score, energy, obs = view
        if x < 0:
            return  # ainda sem posicao valida
        self.current_score = score

        self.tick_count += 1
        obs_l = [o.lower() for o in obs]
        now = time.time()
        if self._threat_seen(obs_l) and not (
                self._steps_only(obs_l) and now < self.ignore_steps_until):
            self.last_threat_at = now

        # impacto: a celula a frente esta bloqueada
        if "blocked" in obs_l:
            vx, vy = DIR_VECTORS.get(d, (0, 0))
            bx, by = x + vx, y + vy
            if in_bounds(bx, by):
                self.world.mark_blocked(bx, by)
                self.log(f"[MAPA] Posicao bloqueada detectada em ({bx},{by})")
            self.path = []  # replanejar

        # travamento: mandamos andar, posicao nao mudou e nao veio 'blocked'
        # (mensagem perdida ou comando ignorado pelo servidor)
        if self.last_action == "forward" and (x, y) == self.last_pos \
                and "blocked" not in obs_l:
            self.forward_fails += 1
            if self.forward_fails >= 3:
                vx, vy = DIR_VECTORS.get(d, (0, 0))
                bx, by = x + vx, y + vy
                if in_bounds(bx, by):
                    self.world.mark_blocked(bx, by)
                    self.log(f"[MAPA] Travado: assumindo bloqueio em ({bx},{by})")
                self.path = []
                self.forward_fails = 0
        elif (x, y) != self.last_pos:
            self.forward_fails = 0
        self.last_pos = (x, y)

        self.world.update_from_observation(x, y, obs)

        # estimador adaptativo de respawn: parado sobre um ponto conhecido,
        # a presenca/ausencia da luz ensina o ritmo de reaparecimento
        if (x, y) in self.world.item_spots and (x, y) in self.last_taken:
            age = time.time() - self.last_taken[(x, y)]
            light_now = any(o.lower() in ("bluelight", "redlight", "weaklight")
                            for o in obs)
            if light_now and age < self.respawn_est:
                self.respawn_est = max(8.0, age)
                self.log(f"[FARM] Respawn mais rapido que o esperado: "
                         f"estimativa -> {self.respawn_est:.0f}s")
            elif not light_now and self.respawn_est < age < 120:
                self.respawn_est = min(60.0, age + 2)

        new_state = self.decide_state(x, y, energy, obs)
        if new_state != self.state:
            self.log(f"[FSM] {self.state} -> {new_state} "
                     f"(pos=({x},{y}) energia={energy} pontos={score} "
                     f"aggr={getattr(self, 'aggr', 0.5):.2f} obs={obs})")
            self.state = new_state
            self.path = []

        handler = {
            "EXPLORE": self.do_explore,
            "GRAB": self.do_grab,
            "ATTACK": self.do_attack,
            "CHASE": self.do_chase,
            "EVADE": self.do_evade,
            "HUNT": self.do_hunt,
            "SURVEY": self.do_survey,
            "RECHARGE": self.do_recharge,
            "FLEE": self.do_flee,
        }[self.state]
        handler(x, y, d, energy, obs)

    def do_grab(self, x, y, d, energy, obs):
        kind = self.world.item_spots.get((x, y), "item")
        self.ai.send_get_item()
        self.last_action = "grab"
        now = time.time()
        self.log(f"[ACAO] Pegar item ({kind}) em ({x},{y})")
        self.world.consume_item(x, y)
        self.last_taken[(x, y)] = now
        self.last_grab[(x, y)] = now
        self.state = "EXPLORE"

    def do_attack(self, x, y, d, energy, obs):
        # economia de municao: tiro custa -10; matar (+1000) exige 10 acertos.
        # Se erramos varios seguidos (inimigo desviando), desengajamos.
        enemy_dist = self.enemy_distance(obs)
        if not self._line_of_fire_clear(x, y, d, enemy_dist):
            self.shots_since_hit = 0
            self.state = "HUNT" if self._combat_first_mode() else "EXPLORE"
            self.log("[COMBATE] Obstaculo conhecido na linha de tiro: segurando fogo")
            return
        if not self._should_attack(energy, enemy_dist, obs):
            obs_l = [o.lower() for o in obs]
            self.state = "EVADE" if "damage" in obs_l else \
                ("HUNT" if self._combat_first_mode() else "EXPLORE")
            return
        if "hit" in [o.lower() for o in obs]:
            self.shots_since_hit = 0
        self.ai.send_shoot()
        self.last_action = "shoot"
        self.shots_since_hit += 1
        dist = enemy_dist if enemy_dist is not None else "?"
        self.log(f"[ACAO] ATIRAR! Inimigo a frente (dist={dist}) energia={energy} "
                 f"tiros_sem_acerto={self.shots_since_hit}")
        if self._combat_first_mode():
            if self.shots_since_hit >= self._miss_limit(enemy_dist):
                self.shots_since_hit = 0
                self.state = "HUNT"
                self.log("[KILLFARM] Alvo saiu da linha: girando para reacquirir"
                         if self._killfarm_mode() else
                         "[KILL] Alvo saiu da linha: girando para reacquirir")
            return
        miss_limit = self._miss_limit(enemy_dist)
        if self.shots_since_hit >= miss_limit:
            pause = 4 if enemy_dist is not None and enemy_dist <= 6 else 6
            self.engage_pause_until = time.time() + pause
            self.shots_since_hit = 0
            self.state = "EXPLORE"
            self.log(f"[FSM] {miss_limit} tiros sem acerto: "
                     f"desengajando por {pause}s (economia)")

    def do_chase(self, x, y, d, energy, obs):
        """Fecha distancia quando ha inimigo na linha, atirando ao ficar perto."""
        enemy_dist = self.enemy_distance(obs)
        if enemy_dist is None:
            self.hunt_turns = CHASE_LOST_HUNT_TURNS
            self.state = "HUNT"
            self.log("[CHASE] Alvo saiu da mira: varrendo para reacquirir")
            self.do_hunt(x, y, d, energy, obs)
            return
        if self._should_attack(energy, enemy_dist, obs):
            self.state = "ATTACK"
            self.do_attack(x, y, d, energy, obs)
            return
        if energy <= LOW_ENERGY:
            self.state = "RECHARGE" if self._due_spots(("powerup",), energy=0) \
                else "EVADE"
            return
        if self._can_chase_forward(x, y, d):
            target = self._front_cell(x, y, d)
            self.ai.send_forward()
            self.last_action = "forward"
            self.log(f"[CHASE] Avancar para fechar distancia "
                     f"(enemy_dist={enemy_dist}) -> {target}")
            return
        self.hunt_turns = CHASE_LOST_HUNT_TURNS
        self.state = "HUNT"
        self.log("[CHASE] Caminho frontal inseguro/bloqueado: girando para rota")
        self.do_hunt(x, y, d, energy, obs)

    def do_evade(self, x, y, d, energy, obs):
        """Sai da linha de tiro antes de caçar ou voltar ao farming."""
        if not self.path:
            target = self._best_evade_cell(x, y, d, obs)
            if target is None:
                self.log("[FSM] Sem celula segura para esquiva imediata")
                self.state = "FLEE" if energy <= LOW_ENERGY else "HUNT"
                return
            self.path = [target]
            self.goal = target
            self.path_allows_flash = False
            self.log(f"[ACAO] Esquivar para {target}")
        self._follow_path(x, y, d)
        if not self.path:
            obs_l = [o.lower() for o in obs]
            self.state = "HUNT" if energy > LOW_ENERGY and "steps" in obs_l else "EXPLORE"

    def do_hunt(self, x, y, d, energy, obs):
        # gira procurando o inimigo. "steps" sozinho pode significar inimigo
        # diagonal/fora da linha: varre uma vez e depois volta a farmar.
        obs_l = [o.lower() for o in obs]
        steps_only = self._steps_only(obs_l)
        if self.hunt_turns <= 0:
            self.hunt_turns = STEP_HUNT_TURNS if (
                self._combat_first_mode() and steps_only
            ) else (8 if self._combat_first_mode() else 4)
        self.ai.send_turn_right()
        self.last_action = "turn"
        self.log(("[KILLFARM] Procurando inimigo (girar a direita)"
                  if self._killfarm_mode() else
                  "[KILL] Procurando inimigo (girar a direita)")
                 if self._combat_first_mode() else
                 "[ACAO] Procurando inimigo (girar a direita)")
        self.hunt_turns -= 1
        if self.hunt_turns == 0:
            self.state = "EXPLORE"
            if self._combat_first_mode() and steps_only:
                self.ignore_steps_until = time.time() + STEP_IGNORE_AFTER_SCAN
                self.log("[KILLFARM] Steps sem alvo na mira: voltando ao farm"
                         if self._killfarm_mode() else
                         "[KILL] Steps sem alvo na mira: reposicionando")

    def do_survey(self, x, y, d, energy, obs):
        if self.survey_turns <= 0:
            self.state = "EXPLORE"
            return
        self.ai.send_turn_right()
        self.last_action = "turn"
        self.survey_turns -= 1
        self.log("[SURVEY] Olhando ao redor do ponto de farm")
        if self.survey_turns == 0:
            self.state = "EXPLORE"

    def do_flee(self, x, y, d, energy, obs):
        # afasta-se: alvo = celula visitada mais distante alcancavel
        if not self.path:
            visited = [(cx, cy) for cx in range(59) for cy in range(34)
                       if self.world.grid[cx][cy] == VISITED and (cx, cy) != (x, y)]
            if visited:
                target, bfs_path = self.world.farthest_reachable((x, y), visited)
                if target is None:
                    return
                self.path = self.world.a_star((x, y), target, start_dir=d) or bfs_path
                self.path_allows_flash = False
                self.log(f"[FSM] Fugindo para {target}")
        self._follow_path(x, y, d)

    def _best_evade_cell(self, x, y, d, obs):
        enemy_visible = self.enemy_distance(obs) is not None
        under_fire = enemy_visible or "damage" in [o.lower() for o in obs]
        candidates = []
        for dir_name, (vx, vy) in DIR_VECTORS.items():
            nx, ny = x + vx, y + vy
            if not in_bounds(nx, ny):
                continue
            cell = self.world.grid[nx][ny]
            if cell == BLOCKED or cell in DANGEROUS:
                continue
            if not self.world.is_walkable(nx, ny):
                continue
            turn_penalty = self.turn_cost(d, dir_name)
            same_axis = (
                (d in ("north", "south") and dir_name in ("north", "south")) or
                (d in ("east", "west") and dir_name in ("east", "west"))
            )
            score = 0
            if under_fire and not same_axis:
                score += 8
            if cell == VISITED:
                score += 3
            score -= turn_penalty
            score -= self.world.pit_risk(nx, ny)
            candidates.append((score, -turn_penalty, (nx, ny)))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][2]

    def do_recharge(self, x, y, d, energy, obs):
        if not self.path:
            powerups = [p for p in self._due_spots(("powerup",), energy=energy)
                        if p != (x, y)]
            goal, bfs_path = self.world.nearest_reachable((x, y), powerups)
            if goal is None:
                self.state = "EXPLORE"
                return
            path = self.world.a_star((x, y), goal, allow_unknown=True,
                                     start_dir=d) or bfs_path
            self.goal = goal
            self.path = path
            self.path_allows_flash = False
            self.log(f"[FSM] Energia baixa ({energy}): indo ao powerup em {goal}")
        self._follow_path(x, y, d)

    def do_explore(self, x, y, d, energy, obs):
        if not self.path:
            if self._kill_mode():
                self._plan_kill_sweep(x, y, d)
            else:
                self._plan_exploration(x, y, d, energy)
            if self.state == "SURVEY":
                self.do_survey(x, y, d, energy, obs)
                return
        self._follow_path(x, y, d)

    def _plan_kill_sweep(self, x, y, d):
        """Modo agressivo: usa exploracao so para varrer mapa e encontrar alvo."""
        start = (x, y)
        frontier = self.world.frontier_cells()
        self.path_allows_flash = False
        if frontier:
            candidates = self._frontier_candidates(start, frontier)
            plan = self._best_scored_plan(
                start, d, candidates,
                lambda target, path, cost: self._frontier_info_gain(target) * 110 - cost)
            if plan is not None:
                _, goal, path, cost = plan
                self.goal = goal
                self.path = path
                self.idle_since = None
                self.log(f"[KILL] Varrendo mapa rumo a {goal} "
                         f"({len(path)} passos, custo={cost})")
                return

        self.state = "HUNT"
        self.hunt_turns = 8
        self.log("[KILL] Sem fronteira: varrendo visao no giro")

    def _plan_exploration(self, x, y, d, energy):
        """Prioridades (estrategia de farming — itens reaparecem):
        1. FARM: ponto de item conhecido provavelmente disponivel;
        2. MODO QUIETO: sem ameaca recente, voltar cedo a ponto de tesouro;
        3. EXPLORAR: fronteira do desconhecido (descobre novos pontos);
        4. ACAMPAR: parar sobre o ponto de tesouro mais 'maduro' e esperar
           o respawn (esperar e gratis; pegar custa 1 acao e rende +1000);
        5. ESPERAR: nada alcancavel — economizar acoes."""
        now = time.time()
        due = [p for p in self._due_spots(("treasure", "unknown", "powerup"),
                                          energy) if p != (x, y)]
        frontier = self.world.frontier_cells()

        start = (x, y)
        self.path_allows_flash = False

        if self._should_defend_current_camp(start, now) and self._quiet_for_farm(now):
            if self._survey_due(now):
                self._start_survey(now, f"Defendendo ponto {start}")
                return
            if self.idle_since is None:
                self.idle_since = now
                wait = max(0.0, self.respawn_est - (now - self.last_taken[start]))
                self.log(f"[FARM] Defendendo ponto {start} "
                         f"(respawn estimado em {wait:.0f}s)")
            return

        if due:
            plan = self._best_scored_plan(
                start, d, due,
                lambda target, path, cost: self._item_value(target, energy, now) - cost)
            if plan is not None:
                _, goal, path, cost = plan
                self.goal = goal
                self.path = path
                self.idle_since = None
                self.log(f"[FARM] Rumo a {goal} ({len(path)} passos, custo={cost})")
                return

        quiet_farm_targets = [p for p in self._farm_camp_targets() if p != start]
        if quiet_farm_targets and self._quiet_for_farm(now):
            plan = self._best_scored_plan(
                start, d, quiet_farm_targets,
                lambda target, path, cost: self._camp_value(target, cost, now))
            if plan is not None:
                _, goal, path, cost = plan
                self.goal = goal
                self.path = path
                self.idle_since = None
                self.log(f"[FARM] Periodo quieto: acampando cedo em {goal} "
                         f"({len(path)} passos, custo={cost})")
                return

        if start in self._farm_camp_targets() and self._quiet_for_farm(now):
            if self._survey_due(now):
                self._start_survey(now, f"Segurando ponto {start}")
                return
            if self.idle_since is None:
                self.idle_since = now
                self.log(f"[FARM] Periodo quieto: segurando ponto {start}")
            return

        if frontier:
            candidates = self._frontier_candidates(start, frontier)
            plan = self._best_scored_plan(
                start, d, candidates,
                lambda target, path, cost: self._frontier_info_gain(target) * 90 - cost * 2)
            if plan is not None:
                _, goal, path, cost = plan
                self.goal = goal
                self.path = path
                self.idle_since = None
                self.log(f"[PLANO] Fronteira promissora {goal} "
                         f"({len(path)} passos, custo={cost})")
                return

        # sem rota segura para nada: atravessar suspeita APENAS de teleporte
        # (teleporte nao mata; poco continua proibido)
        all_targets = due + frontier
        if all_targets:
            if frontier:
                all_targets = due + self._frontier_candidates(start, frontier)
            plan = self._best_scored_plan(
                start, d, all_targets,
                lambda target, path, cost: (
                    self._item_value(target, energy, now)
                    if target in self.world.item_spots
                    else self._frontier_info_gain(target) * 70
                ) - cost * 3,
                allow_flash=True)
            if plan is not None:
                _, goal, path, cost = plan
                self.goal = goal
                self.path = path
                self.path_allows_flash = True
                self.idle_since = None
                self.log(f"[PLANO] Sem rota segura: arriscando teleporte rumo a {goal} "
                         f"(custo={cost})")
                return

        # acampar: ir ao ponto de tesouro coletado ha mais tempo (o proximo
        # a reaparecer) e esperar em cima dele
        spots = [p for p, k in self.world.item_spots.items()
                 if k in ("treasure", "unknown")]
        if spots:
            camp = min(spots, key=lambda p: self.last_taken.get(p, 0))
            if camp == (x, y):
                if self.idle_since is None:
                    self.idle_since = now
                    self.log(f"[FARM] Acampando em {camp} a espera do respawn")
                return
            goal, bfs_path = self.world.nearest_reachable((x, y), [camp])
            if goal:
                path = self.world.a_star((x, y), goal, allow_unknown=True,
                                         start_dir=d) or bfs_path
                self.goal = goal
                self.path = path
                self.idle_since = None
                self.log(f"[FARM] Indo acampar em {goal}")
                return

        # nada alcancavel: FICAR PARADO (acoes custam -1; esperar e gratis).
        # Se o impasse durar, arrisca a celula desconhecida de menor risco
        # logico para destravar o mapa.
        if self.idle_since is None:
            self.idle_since = now
            self.log("[PLANO] Sem alvos alcancaveis: economizando acoes")
        elif now - self.idle_since > 15:
            self.idle_since = now
            unknown = self._unknown_neighbors(x, y)
            if unknown:
                best = min(unknown, key=lambda n: self.world.pit_risk(*n))
                if self.world.pit_risk(*best) <= 1:
                    self.path = [best]
                    self.log(f"[PLANO] Impasse: arriscando {best} "
                             f"(risco={self.world.pit_risk(*best)})")

    def _unknown_neighbors(self, x, y):
        from world_model import UNKNOWN
        return [n for n in neighbors(x, y) if self.world.grid[n[0]][n[1]] == UNKNOWN]

    def _follow_path(self, x, y, d):
        if not self.path:
            return
        # descarta celulas ja alcancadas
        if self.path[0] == (x, y):
            self.path.pop(0)
            if not self.path:
                return
        target = self.path[0]
        # caminho invalido (nao adjacente, ex: apos teleporte) -> replaneja
        if abs(target[0] - x) + abs(target[1] - y) != 1:
            self.log("[PLANO] Caminho invalido (teleporte?): replanejando")
            self.path = []
            return
        # alvo ficou perigoso/bloqueado com novo conhecimento -> replaneja.
        # Celula suspeita de poco NUNCA e pisada; suspeita de teleporte so
        # quando o plano atual foi feito explicitamente com allow_flash.
        cell = self.world.grid[target[0]][target[1]]
        flash_ok = self.path_allows_flash and cell == DANGER_FLASH
        if cell == BLOCKED or (cell in DANGEROUS and not flash_ok):
            self.log(f"[PLANO] Proxima celula {target} ficou insegura: replanejando")
            self.path = []
            return
        before = (x, y)
        self.step_towards(x, y, d, target)
        if before == target:
            self.path.pop(0)
