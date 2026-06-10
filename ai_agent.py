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
                         in_bounds, neighbors)
from fuzzy import combat_aggressiveness

LOW_ENERGY = 40  # abaixo disso, busca powerup conhecido

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
        # deteccao de travamento (forward sem efeito e sem 'blocked')
        self.last_pos = None
        self.last_action = None
        self.forward_fails = 0
        self.sync_fails = 0
        # economia de combate: desengaja apos varios tiros sem acerto
        self.shots_since_hit = 0
        self.engage_pause_until = 0.0
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
            elif self.aggr >= 0.35:
                return "ATTACK"
            else:
                return "FLEE"
        if took_damage:
            # levamos tiro: o atirador esta em linha reta conosco
            return "HUNT" if self.aggr >= 0.5 else "FLEE"
        if hears_steps and self.aggr < 0.25:
            return "FLEE"
        if energy <= LOW_ENERGY and self._due_spots(("powerup",), energy=0):
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
            if kind == "powerup" and energy > 70:
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

        self.tick_count += 1

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
            "HUNT": self.do_hunt,
            "RECHARGE": self.do_recharge,
            "FLEE": self.do_flee,
        }[self.state]
        handler(x, y, d, energy, obs)

    def do_grab(self, x, y, d, energy, obs):
        kind = self.world.item_spots.get((x, y), "item")
        self.ai.send_get_item()
        now = time.time()
        self.log(f"[ACAO] Pegar item ({kind}) em ({x},{y})")
        self.world.consume_item(x, y)
        self.last_taken[(x, y)] = now
        self.last_grab[(x, y)] = now
        self.state = "EXPLORE"

    def do_attack(self, x, y, d, energy, obs):
        # economia de municao: tiro custa -10; matar (+1000) exige 10 acertos.
        # Se erramos varios seguidos (inimigo desviando), desengajamos.
        if "hit" in obs:
            self.shots_since_hit = 0
        self.ai.send_shoot()
        self.shots_since_hit += 1
        dist = next((o.split("#")[1] for o in obs
                     if o.startswith(("enemy", "eneny")) and "#" in o), "?")
        self.log(f"[ACAO] ATIRAR! Inimigo a frente (dist={dist}) energia={energy} "
                 f"tiros_sem_acerto={self.shots_since_hit}")
        if self.shots_since_hit >= 5:
            self.engage_pause_until = time.time() + 6
            self.shots_since_hit = 0
            self.state = "EXPLORE"
            self.log("[FSM] 5 tiros sem acerto: desengajando por 6s (economia)")

    def do_hunt(self, x, y, d, energy, obs):
        # gira procurando o inimigo; se der 4 voltas sem achar, volta a explorar
        if self.hunt_turns <= 0:
            self.hunt_turns = 4
        self.ai.send_turn_right()
        self.log("[ACAO] Procurando inimigo (girar a direita)")
        self.hunt_turns -= 1
        if self.hunt_turns == 0:
            self.state = "EXPLORE"

    def do_flee(self, x, y, d, energy, obs):
        # afasta-se: alvo = celula visitada mais distante alcancavel
        if not self.path:
            visited = [(cx, cy) for cx in range(59) for cy in range(34)
                       if self.world.grid[cx][cy] == VISITED and (cx, cy) != (x, y)]
            if visited:
                target = max(visited, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
                self.path = self.world.a_star((x, y), target, start_dir=d) or []
                self.path_allows_flash = False
                self.log(f"[FSM] Fugindo para {target}")
        self._follow_path(x, y, d)

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
        """Prioridades (estrategia de farming — itens reaparecem):
        1. FARM: ponto de item conhecido provavelmente disponivel;
        2. EXPLORAR: fronteira do desconhecido (descobre novos pontos);
        3. ACAMPAR: parar sobre o ponto de tesouro mais 'maduro' e esperar
           o respawn (esperar e gratis; pegar custa 1 acao e rende +1000);
        4. ESPERAR: nada alcancavel — economizar acoes."""
        now = time.time()
        due = [p for p in self._due_spots(("treasure", "unknown", "powerup"),
                                          energy) if p != (x, y)]
        frontier = self.world.frontier_cells()

        self.path_allows_flash = False
        for targets, label in ((due, "FARM"), (frontier, "PLANO")):
            if not targets:
                continue
            # BFS multi-alvo: garante achar QUALQUER alvo alcancavel
            goal, bfs_path = self.world.nearest_reachable((x, y), targets)
            if goal:
                # A* refina o caminho minimizando giros (mesma conectividade)
                path = self.world.a_star((x, y), goal, allow_unknown=True,
                                         start_dir=d) or bfs_path
                self.goal = goal
                self.path = path
                self.idle_since = None
                self.log(f"[{label}] Rumo a {goal} ({len(path)} passos)")
                return

        # sem rota segura para nada: atravessar suspeita APENAS de teleporte
        # (teleporte nao mata; poco continua proibido)
        all_targets = due + frontier
        if all_targets:
            goal, bfs_path = self.world.nearest_reachable((x, y), all_targets,
                                                          allow_flash=True)
            if goal:
                path = self.world.a_star((x, y), goal, allow_unknown=True,
                                         start_dir=d, allow_flash=True) or bfs_path
                self.goal = goal
                self.path = path
                self.path_allows_flash = True
                self.idle_since = None
                self.log(f"[PLANO] Sem rota segura: arriscando teleporte rumo a {goal}")
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
