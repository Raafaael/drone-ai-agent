"""
Inteligencia do drone: Maquina de Estados Finitos + inferencia logica
(world_model) + busca A* + logica fuzzy para decisoes de combate.

Estados:
  EXPLORE  - explora a fronteira do mapa conhecido em busca de itens;
  GRAB     - pega item na celula atual (luz detectada);
  ATTACK   - inimigo na mira e agressividade fuzzy alta -> atira;
  HUNT     - levou dano (atirador em linha reta) -> gira procurando-o;
  RECHARGE - energia baixa -> vai ate um powerup conhecido;
  FLEE     - agressividade fuzzy baixa (fraco/ameacado) -> foge.

A transicao ATTACK/HUNT/FLEE usa a 'agressividade' calculada por um
controlador fuzzy (fuzzy.py) sobre energia e distancia do inimigo.
"""

import time

from world_model import (WorldModel, DIR_VECTORS, DANGEROUS, DANGER_PIT,
                         DANGER_BOTH, DANGER_FLASH, VISITED, BLOCKED,
                         UNKNOWN, in_bounds, neighbors)
from fuzzy import combat_aggressiveness

LOW_ENERGY = 55  # abaixo disso, busca powerup conhecido com mais urgencia
FARM_MIN_VISITED = 45       # farm mais cedo quando ja ha pontos bons
LOOP_WINDOW = 26            # janela para detectar que estamos rodando localmente
LOOP_BOX_AREA = 24
MAX_ATTACK_DIST = 6         # tiro distante demais costuma virar custo puro
DEFAULT_TREASURE_VALUE = 950
DEFAULT_UNKNOWN_VALUE = 450

TURN_LEFT_OF = {"north": "west", "west": "south", "south": "east", "east": "north"}
TURN_RIGHT_OF = {"north": "east", "east": "south", "south": "west", "west": "north"}


class DroneAgent:
    def __init__(self, game_ai, log=print):
        self.ai = game_ai
        self.world = WorldModel()
        self.log = log
        self.state = "EXPLORE"
        self.path = []            # caminho atual (lista de celulas)
        self.path_allows_flash = False
        self.goal = None
        self.hunt_turns = 0       # giros restantes no estado HUNT
        self.last_taken = {}      # (x,y) -> timestamp da ultima coleta
        self.tick_count = 0
        self.collect_count = 0
        self.collect_by_kind = {}
        # deteccao de travamento (forward sem efeito e sem 'blocked')
        self.last_pos = None
        self.last_action = None
        self.forward_fails = 0
        self.sync_fails = 0
        # economia de combate: desengaja apos varios tiros sem acerto
        self.shots_since_hit = 0
        self.shots_fired = 0
        self.shots_hit = 0
        self.awaiting_shot_result = False
        self.engage_pause_until = 0.0
        # economia de acoes: instante em que ficamos sem alvos
        self.idle_since = None
        # farming: estimativa adaptativa do tempo de respawn dos itens e
        # guarda anti-spam de 'pegar' por celula
        self.respawn_est = 3.5
        self.last_grab = {}
        self.pending_grab = None
        self.item_values = {}        # (x,y) -> ganho liquido observado
        self.spot_respawn = {}       # (x,y) -> estimativa local
        self.spot_next_check = {}    # (x,y) -> nao voltar antes disso
        self.spot_misses = {}        # visitas a ponto ainda sem luz
        # memoria tatica para reduzir loops e usar teleporte com controle
        self.seen_cells = set()
        self.recent_positions = []
        self.last_target = None
        self.teleport_landings = {}

    # ---------------- percepcao ----------------

    def sync(self):
        """Atualiza posicao/energia e observacoes do servidor
        (status + observacao pedidos em paralelo). O timeout cresce com
        falhas consecutivas: servidor lento atrasa o agente, mas nunca o
        faz agir com dados velhos."""
        timeout = 0.4 + min(self.sync_fails * 0.2, 1.6)
        return self.ai.request_sync_pair(timeout=timeout)

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
            self.last_action = "wait"
            self.last_target = None
            return  # celula nao adjacente: caminho sera recalculado
        if self.face(d, want):
            self.ai.send_forward()
            self.last_action = "forward"
            self.last_target = target
            self.log(f"[ACAO] Andar para frente -> {target}")
        else:
            self.last_action = "turn"
            self.last_target = None

    # ---------------- decisao (FSM) ----------------

    def decide_state(self, x, y, energy, obs):
        # o servidor pode enviar "enemy#xx" ou "eneny#xx" (grafia do enunciado)
        enemy_dist = None
        for o in obs:
            if o.startswith(("enemy", "eneny")):
                try:
                    enemy_dist = int(o.split("#")[1])
                except (IndexError, ValueError):
                    enemy_dist = 5
        took_damage = "damage" in obs
        hears_steps = "steps" in obs

        # a LUZ observada e a verdade do servidor: se ha luz, ha item AGORA
        # (essencial para farming de respawns). Guarda anti-spam evita pegar
        # duas vezes antes de o servidor processar.
        obs_l = [o.lower() for o in obs]
        if "bluelight" in obs_l:
            light = "treasure"
        elif "redlight" in obs_l:
            light = "powerup"
        elif "weaklight" in obs_l:
            light = "unknown"
        else:
            light = None
        has_item = light is not None and \
            (light != "powerup" or energy <= 70) and \
            time.time() - self.last_grab.get((x, y), 0) > 0.6

        # agressividade fuzzy: energia x distancia do inimigo. Sem inimigo
        # visivel, dano/steps implicam inimigo perto (dist ~2); senao longe.
        if enemy_dist is not None:
            assumed_dist = enemy_dist
        elif took_damage or hears_steps:
            assumed_dist = 2
        else:
            assumed_dist = 10
        self.aggr = combat_aggressiveness(energy, assumed_dist)

        # item embaixo do drone vale sempre: pegar custa 1 acao
        if has_item:
            return "GRAB"
        if enemy_dist is not None:
            # desengajado por excesso de tiros errados? ignora e segue
            if time.time() < self.engage_pause_until and "hit" not in obs:
                pass
            elif self._should_attack(enemy_dist, energy):
                return "ATTACK"
            elif enemy_dist <= 3 or self.aggr < 0.30:
                return "FLEE"
            else:
                return "EXPLORE"
        if took_damage:
            # levamos tiro: o atirador esta em linha reta conosco
            return "HUNT" if self.aggr >= 0.5 else "FLEE"
        if hears_steps and self.aggr < 0.25:
            return "FLEE"
        if energy <= LOW_ENERGY and self._due_spots(("powerup",), energy=0):
            return "RECHARGE"
        return "EXPLORE"

    def _should_attack(self, enemy_dist, energy):
        """Tiro custa caro; so insistimos quando a chance/beneficio compensa."""
        if time.time() < self.engage_pause_until:
            return False
        if enemy_dist > MAX_ATTACK_DIST and energy < 85:
            return False
        if energy < 35 and enemy_dist > 2:
            return False

        hit_rate = self.shots_hit / self.shots_fired if self.shots_fired else 0.5
        if self.shots_fired >= 6 and hit_rate < 0.20 and enemy_dist > 3:
            return False
        if self.shots_since_hit >= 3 and enemy_dist > 4:
            return False
        return self.aggr >= 0.42 or (energy >= 80 and enemy_dist <= 7)

    def _due_spots(self, kinds, energy):
        """Pontos de item da memoria permanente provavelmente disponiveis:
        nunca coletados, ou coletados ha mais que a estimativa de respawn.
        Powerups so contam com energia <= 70 (senao seriam desperdicados)."""
        now = time.time()
        out = []
        for pos, kind in self.world.item_spots.items():
            if kind not in kinds:
                continue
            if kind == "powerup" and energy > 70:
                continue
            if now < self.spot_next_check.get(pos, 0):
                continue
            estimate = self.spot_respawn.get(pos, self.respawn_est)
            if pos not in self.last_taken or now - self.last_taken[pos] >= estimate:
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

        self.tick_count += 1
        prev_pos = self.last_pos

        if self.pending_grab:
            pos, prev_score, grabbed_at = self.pending_grab
            if score != prev_score or time.time() - grabbed_at > 1.2:
                delta = score - prev_score
                if delta > 0:
                    old = self.item_values.get(pos)
                    self.item_values[pos] = delta if old is None else int(old * 0.7 + delta * 0.3)
                    self.log(f"[FARM] Valor aprendido em {pos}: +{self.item_values[pos]}")
                self.pending_grab = None

        if self.awaiting_shot_result:
            if "hit" in obs:
                self.shots_hit += 1
                self.shots_since_hit = 0
            else:
                self.shots_since_hit += 1
            self.awaiting_shot_result = False

        if self.last_action == "forward" and prev_pos and (x, y) != prev_pos:
            jumped = abs(x - prev_pos[0]) + abs(y - prev_pos[1]) > 1
            if jumped and self.last_target:
                landings = self.teleport_landings.setdefault(self.last_target, {})
                landings[(x, y)] = landings.get((x, y), 0) + 1
                self.log(f"[MAPA] Teleporte provavel em {self.last_target} -> ({x},{y})")

        if not self.recent_positions or self.recent_positions[-1] != (x, y):
            self.recent_positions.append((x, y))
            if len(self.recent_positions) > LOOP_WINDOW:
                self.recent_positions.pop(0)

        # impacto: a celula a frente esta bloqueada
        if "blocked" in obs:
            vx, vy = DIR_VECTORS.get(d, (0, 0))
            bx, by = x + vx, y + vy
            if in_bounds(bx, by):
                self.world.mark_blocked(bx, by)
                self.log(f"[MAPA] Posicao bloqueada detectada em ({bx},{by})")
            self.path = []  # replanejar

        # travamento: mandamos andar, posicao nao mudou e nao veio 'blocked'
        # (mensagem perdida ou comando ignorado pelo servidor)
        if self.last_action == "forward" and (x, y) == self.last_pos \
                and "blocked" not in obs:
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
        self.seen_cells.add((x, y))

        # estimador adaptativo de respawn: parado sobre um ponto conhecido,
        # a presenca/ausencia da luz ensina o ritmo de reaparecimento
        if (x, y) in self.world.item_spots and (x, y) in self.last_taken:
            age = time.time() - self.last_taken[(x, y)]
            light_now = any(o.lower() in ("bluelight", "redlight", "weaklight")
                            for o in obs)
            if light_now and age < self.respawn_est:
                self.respawn_est = max(2.0, age)
                self.spot_respawn[(x, y)] = max(2.0, age)
                self.spot_misses[(x, y)] = 0
                self.spot_next_check[(x, y)] = time.time()
                self.log(f"[FARM] Respawn mais rapido que o esperado: "
                         f"estimativa -> {self.respawn_est:.0f}s")
            elif not light_now and self.respawn_est < age < 120:
                self.respawn_est = min(45.0, age + 1.5)
                misses = self.spot_misses.get((x, y), 0) + 1
                self.spot_misses[(x, y)] = misses
                estimate = self.spot_respawn.get((x, y), self.respawn_est)
                self.spot_respawn[(x, y)] = min(60.0, max(estimate, age + 1.5))
                self.spot_next_check[(x, y)] = time.time() + min(8.0, 1.5 * misses)

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
            "HUNT": self.do_hunt,
            "RECHARGE": self.do_recharge,
            "FLEE": self.do_flee,
        }[self.state]
        handler(x, y, d, energy, obs)

    def do_grab(self, x, y, d, energy, obs):
        kind = self.world.item_spots.get((x, y), "item")
        prev_score = self.ai.score
        self.ai.send_get_item()
        self.last_action = "grab"
        self.last_target = None
        now = time.time()
        self.world.consume_item(x, y)
        self.last_taken[(x, y)] = now
        self.last_grab[(x, y)] = now
        self.collect_count += 1
        self.collect_by_kind[kind] = self.collect_by_kind.get(kind, 0) + 1
        self.pending_grab = ((x, y), prev_score, now)
        self.spot_next_check[(x, y)] = now + self.spot_respawn.get((x, y), self.respawn_est)
        self.spot_misses[(x, y)] = 0
        self.log(f"[COLETA] {kind} em ({x},{y}) | total_coletas={self.collect_count}")
        self.state = "EXPLORE"

    def do_attack(self, x, y, d, energy, obs):
        # economia de municao: tiro custa -10; matar (+1000) exige 10 acertos.
        # Se erramos varios seguidos (inimigo desviando), desengajamos.
        self.ai.send_shoot()
        self.last_action = "shoot"
        self.last_target = None
        self.shots_fired += 1
        self.awaiting_shot_result = True
        dist = next((o.split("#")[1] for o in obs
                     if o.startswith(("enemy", "eneny")) and "#" in o), "?")
        self.log(f"[ACAO] ATIRAR! Inimigo a frente (dist={dist}) energia={energy} "
                 f"tiros_sem_acerto={self.shots_since_hit}")
        if self.shots_since_hit >= 4:
            self.engage_pause_until = time.time() + 8
            self.shots_since_hit = 0
            self.state = "EXPLORE"
            self.log("[FSM] Muitos tiros sem acerto: desengajando por 8s")

    def do_hunt(self, x, y, d, energy, obs):
        # gira procurando o inimigo; se der 4 voltas sem achar, volta a explorar
        if self.hunt_turns <= 0:
            self.hunt_turns = 4
        self.ai.send_turn_right()
        self.last_action = "turn"
        self.last_target = None
        self.log("[ACAO] Procurando inimigo (girar a direita)")
        self.hunt_turns -= 1
        if self.hunt_turns == 0:
            self.state = "EXPLORE"

    def do_flee(self, x, y, d, energy, obs):
        # afasta-se para uma celula visitada com saidas, pouco revisitada e
        # longe do confronto; distancia pura costuma escolher becos.
        if not self.path:
            visited = [(cx, cy) for cx in range(59) for cy in range(34)
                       if self.world.grid[cx][cy] == VISITED and (cx, cy) != (x, y)]
            if visited:
                scored = sorted(visited, key=lambda c: self._flee_score(c, x, y),
                                reverse=True)[:30]
                for target in scored:
                    path = self.world.a_star((x, y), target, start_dir=d)
                    if path:
                        self.path = path
                        self.path_allows_flash = False
                        self.log(f"[FSM] Fugindo para {target}")
                        break
        self._follow_path(x, y, d)

    def _flee_score(self, cell, x, y):
        dist = abs(cell[0] - x) + abs(cell[1] - y)
        exits = self.world.safe_exit_count(*cell)
        revisits = self.world.visit_count.get(cell, 0)
        power_bonus = 0
        for pos, kind in self.world.item_spots.items():
            if kind == "powerup":
                power_bonus = max(power_bonus, 12 - abs(pos[0] - cell[0]) - abs(pos[1] - cell[1]))
        return dist * 2 + exits * 8 + power_bonus - revisits * 3

    def do_recharge(self, x, y, d, energy, obs):
        if not self.path:
            powerups = [p for p in self._due_spots(("powerup",), energy=0)
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
            self._plan_exploration(x, y, d, energy)
        self._follow_path(x, y, d)

    def _plan_exploration(self, x, y, d, energy):
        """Escolhe objetivos por ganho esperado, nao so por distancia.

        Inicio da partida privilegia abrir mapa; farming cresce quando ja
        existe massa critica de celulas visitadas ou quando o tesouro esta
        barato. Impasses/loops liberam teleporte antes de ficar parado."""
        now = time.time()
        due = [p for p in self._due_spots(("treasure", "unknown", "powerup"),
                                          energy) if p != (x, y)]
        frontier = self.world.frontier_cells()
        phase = self._strategy_phase()
        looping = self._looping_locally()

        self.path_allows_flash = False

        if due and (phase != "EARLY_EXPLORE" or looping or not frontier):
            if self._plan_to_scored_targets(x, y, d, due, "FARM", energy):
                return

        if due and phase == "EARLY_EXPLORE" and frontier:
            # No comeco, so desvia para farm se estiver praticamente no caminho.
            if self._plan_to_scored_targets(x, y, d, due, "FARM", energy,
                                            max_path_len=7):
                return

        if frontier:
            if self._plan_to_scored_targets(x, y, d, frontier, "PLANO", energy):
                return

        if due:
            if self._plan_to_scored_targets(x, y, d, due, "FARM", energy):
                return

        # sem rota segura para nada: atravessar suspeita APENAS de teleporte
        # (teleporte nao mata; poco continua proibido)
        all_targets = due + frontier
        stuck_time = 0 if self.idle_since is None else now - self.idle_since
        if all_targets and (looping or stuck_time > 8):
            if self._plan_to_scored_targets(x, y, d, all_targets, "TELE", energy,
                                            allow_flash=True):
                return

        # acampar: ir ao ponto de tesouro coletado ha mais tempo (o proximo
        # a reaparecer) e esperar em cima dele
        spots = [p for p, k in self.world.item_spots.items()
                 if k in ("treasure", "unknown")]
        if spots and phase != "EARLY_EXPLORE":
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
        elif now - self.idle_since > 10:
            self.idle_since = now
            unknown = self._unknown_neighbors(x, y)
            if unknown:
                best = min(unknown, key=lambda n: self.world.pit_risk(*n))
                if self.world.pit_risk(*best) <= (1 if looping else 0):
                    self.path = [best]
                    self.log(f"[PLANO] Impasse: arriscando {best} "
                             f"(risco={self.world.pit_risk(*best)})")

    def _strategy_phase(self):
        treasure_spots = sum(1 for kind in self.world.item_spots.values()
                             if kind in ("treasure", "unknown"))
        if treasure_spots >= 2:
            return "FARM"
        if len(self.seen_cells) < FARM_MIN_VISITED:
            return "EARLY_EXPLORE"
        if len(self.world.item_spots) >= 3:
            return "FARM"
        return "EXPAND"

    def _looping_locally(self):
        if len(self.recent_positions) < LOOP_WINDOW:
            return False
        xs = [p[0] for p in self.recent_positions]
        ys = [p[1] for p in self.recent_positions]
        area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        unique = len(set(self.recent_positions))
        return area <= LOOP_BOX_AREA and unique <= LOOP_WINDOW // 2

    def _plan_to_scored_targets(self, x, y, d, targets, label, energy,
                                allow_flash=False, max_path_len=None):
        ranked = sorted(set(targets), key=lambda c: self._target_score(c, x, y, energy),
                        reverse=True)[:45]
        best = None
        for target in ranked:
            path = self.world.a_star((x, y), target, allow_unknown=True,
                                     start_dir=d, allow_flash=allow_flash)
            if path is None:
                continue
            if max_path_len is not None and len(path) > max_path_len:
                continue
            score = self._target_score(target, x, y, energy) - len(path) * 2.2
            if best is None or score > best[0]:
                best = (score, target, path)
        if best is None:
            return False
        _, goal, path = best
        self.goal = goal
        self.path = path
        self.path_allows_flash = allow_flash
        self.idle_since = None
        tag = "PLANO" if label == "PLANO" else label
        extra = " com teleporte" if allow_flash else ""
        self.log(f"[{tag}] Rumo a {goal}{extra} ({len(path)} passos)")
        return True

    def _target_score(self, cell, x, y, energy):
        dist = abs(cell[0] - x) + abs(cell[1] - y)
        grid_cell = self.world.grid[cell[0]][cell[1]]
        revisits = self.world.visit_count.get(cell, 0)
        recent_penalty = 18 if cell in self.recent_positions else 0
        pit_penalty = self.world.pit_risk(*cell) * 55
        tele_penalty = self.world.teleport_risk(*cell) * 9
        exits = self.world.safe_exit_count(*cell)

        if cell in self.world.item_spots:
            kind = self.world.item_spots[cell]
            age = time.time() - self.last_taken.get(cell, 0)
            estimate = self.spot_respawn.get(cell, self.respawn_est)
            maturity = min(age / max(estimate, 1.0), 1.5)
            learned = self.item_values.get(cell)
            if kind == "treasure":
                value = (learned or DEFAULT_TREASURE_VALUE) * (0.45 + 0.75 * maturity)
            elif kind == "powerup":
                value = 260 if energy <= 55 else (120 if energy <= 75 else 10)
            else:
                value = (learned or DEFAULT_UNKNOWN_VALUE) * (0.35 + 0.70 * maturity)
            if time.time() < self.spot_next_check.get(cell, 0):
                value *= 0.10
            value -= self.spot_misses.get(cell, 0) * 35
        else:
            unknown_neighbors = sum(1 for n in neighbors(*cell)
                                    if self.world.grid[n[0]][n[1]] == UNKNOWN)
            value = unknown_neighbors * 28 + exits * 9
            if grid_cell == UNKNOWN:
                value += 18

        return value - dist * 1.2 - revisits * 5 - recent_penalty \
            - pit_penalty - tele_penalty

    def _unknown_neighbors(self, x, y):
        from world_model import UNKNOWN
        return [n for n in neighbors(x, y) if self.world.grid[n[0]][n[1]] == UNKNOWN]

    def _follow_path(self, x, y, d):
        if not self.path:
            self.last_action = "wait"
            self.last_target = None
            return
        # descarta celulas ja alcancadas
        if self.path[0] == (x, y):
            self.path.pop(0)
            if not self.path:
                self.last_action = "wait"
                self.last_target = None
                return
        target = self.path[0]
        # caminho invalido (nao adjacente, ex: apos teleporte) -> replaneja
        if abs(target[0] - x) + abs(target[1] - y) != 1:
            self.log("[PLANO] Caminho invalido (teleporte?): replanejando")
            self.path = []
            self.last_action = "wait"
            self.last_target = None
            return
        # alvo ficou perigoso/bloqueado com novo conhecimento -> replaneja.
        # Celula suspeita de poco NUNCA e pisada; suspeita de teleporte so
        # quando o plano atual foi feito explicitamente com allow_flash.
        cell = self.world.grid[target[0]][target[1]]
        flash_ok = self.path_allows_flash and cell == DANGER_FLASH
        if cell == BLOCKED or (cell in DANGEROUS and not flash_ok):
            self.log(f"[PLANO] Proxima celula {target} ficou insegura: replanejando")
            self.path = []
            self.last_action = "wait"
            self.last_target = None
            return
        before = (x, y)
        self.step_towards(x, y, d, target)
        if before == target:
            self.path.pop(0)
