"""
Testes offline do agente (sem servidor real):
1. Testes do modelo de mundo (inferencia + A*).
2. Simulador local minimal do GameServer para exercitar o loop completo.

Uso: python tests/test_offline.py
"""

import os
import socket
import sys
import threading
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from world_model import WorldModel, SAFE, VISITED, BLOCKED, DANGEROUS


def test_world_model():
    w = WorldModel()
    # visita (5,5) sem brisa/flash -> vizinhos seguros
    w.update_from_observation(5, 5, [])
    assert w.grid[5][5] == VISITED
    assert w.grid[5][4] == SAFE and w.grid[6][5] == SAFE

    # brisa em (10,10) -> vizinhos desconhecidos viram suspeitos de poco
    w.update_from_observation(10, 10, ["breeze"])
    assert w.grid[10][9] in DANGEROUS

    # mas se (10,9) for visitada depois, deixa de ser suspeita
    w.update_from_observation(10, 9, [])
    assert w.grid[10][9] == VISITED

    # item detectado
    w.update_from_observation(7, 7, ["blueLight"])
    assert w.items[(7, 7)] == "treasure"

    # A* simples entre celulas seguras
    for i in range(5, 9):
        w.update_from_observation(i, 5, [])
    path = w.a_star((5, 5), (8, 5))
    assert path == [(6, 5), (7, 5), (8, 5)], path

    # A* desvia de bloqueio
    w.mark_blocked(7, 5)
    path = w.a_star((5, 5), (8, 5))
    assert path and (7, 5) not in path

    # A* NUNCA aceita celula suspeita de poco, nem como objetivo
    w2 = WorldModel()
    w2.update_from_observation(10, 10, ["breeze"])
    assert w2.grid[10][9] in DANGEROUS
    assert w2.a_star((10, 10), (10, 9)) is None

    # suspeita apenas de teleporte: proibida por padrao, permitida com
    # allow_flash (ultimo recurso)
    w3 = WorldModel()
    w3.update_from_observation(20, 20, ["flash"])
    assert w3.a_star((20, 20), (20, 19)) is None
    assert w3.a_star((20, 20), (20, 19), allow_flash=True) is not None

    # BFS multi-alvo encontra o alvo alcancavel mais proximo
    w5 = WorldModel()
    for i in range(5):
        w5.update_from_observation(5 + i, 5, [])
    goal, path = w5.nearest_reachable((5, 5), [(9, 5), (30, 30)])
    assert goal == (9, 5) and path[-1] == (9, 5)

    # resolucao logica: brisa com um unico candidato -> poco CONFIRMADO
    w4 = WorldModel()
    w4.update_from_observation(10, 10, ["breeze"])
    w4.update_from_observation(10, 9, [])    # exonera
    w4.update_from_observation(11, 10, [])   # exonera
    w4.update_from_observation(10, 11, [])   # exonera
    assert (9, 10) in w4.confirmed_pits, w4.confirmed_pits
    assert w4.pit_risk(9, 10) == 99
    assert w4.pit_risk(10, 9) == 0  # visitada, sem risco

    print("OK: world_model (inferencia + resolucao logica + A*)")


def test_fuzzy():
    from fuzzy import combat_aggressiveness as aggr
    assert aggr(95, 3) > 0.8, "forte e perto deve ser muito agressivo"
    assert aggr(15, 2) < 0.25, "fraco e perto deve fugir"
    assert aggr(15, 2) < aggr(15, 10), "fraco: quanto mais perto, menos agressivo"
    assert aggr(95, 3) > aggr(50, 3), "mais energia -> mais agressivo"
    for e in range(0, 101, 10):
        for dist in range(1, 11):
            v = aggr(e, dist)
            assert 0.0 <= v <= 1.0
    print("OK: fuzzy (agressividade de combate)")


def test_observation_normalization():
    from observations import Observation, normalize_token

    obs = Observation.from_tokens([
        " blueLight ", "eneny#3", "h", "d", "steps", "", "greenLight",
        "enemy#bad",
    ])
    assert "bluelight" in obs.tokens
    assert "enemy#3" in obs.tokens
    assert "hit" in obs.tokens and "damage" in obs.tokens
    assert obs.enemy_distance == 3
    assert obs.light == "treasure"
    assert obs.steps
    assert normalize_token("enemy#bad") == "enemy#5"
    assert Observation.from_tokens(["weakLight"]).light == "unknown"
    assert Observation.from_tokens(["redLight"]).light == "powerup"
    assert Observation.from_tokens(["blocked"]).blocked
    print("OK: observations (normalizacao e representacao estruturada)")


def test_async_notifications_are_preserved():
    from communication import GameAI

    ai = GameAI(log=lambda m: None)
    ai._handle_message(["notification", "damage"])
    ai._handle_message(["h"])
    assert "damage" in ai.observations and "hit" in ai.observations
    ai._handle_message(["o", "enemy#3"])
    assert "enemy#3" in ai.observations
    assert "damage" in ai.observations and "hit" in ai.observations
    assert ai.pending_observations == []
    ai._handle_message(["o", ""])
    assert "damage" not in ai.observations and "hit" not in ai.observations
    ai._handle_message(["u", "p1", "100"])
    assert ai.request_scoreboard_sync(timeout=0.01) is None
    assert ai.scoreboard == ["p1", "100"]
    print("OK: communication (hit/damage assincronos preservados)")


def test_world_model_extended_features():
    from observations import Observation

    w = WorldModel()
    w.update_pose(5, 5, "east")
    assert w.position == (5, 5) and w.orientation == "east"
    w.update_from_observation(5, 5, Observation.from_tokens([]))
    w.update_from_observation(5, 5, Observation.from_tokens([]))
    assert w.visit_count[(5, 5)] == 2
    w.update_from_observation(10, 10, Observation.from_tokens(["flash"]))
    assert w.teleport_risk(10, 9) > 0
    assert w.safe_exit_count(5, 5) >= 1
    w.mark_blocked(6, 5)
    assert w.grid[6][5] == BLOCKED
    old_version = w.version
    w.reset()
    assert w.version == 0 and w.position is None
    assert all(cell == "?" for col in w.grid for cell in col)
    assert old_version > 0
    print("OK: world_model estendido (pose, visitas, teleporte, reset)")


class MiniServer(threading.Thread):
    """Simulador minimo do GameServer (1 cliente) para teste de fumaca."""

    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port
        self.x, self.y, self.dir = 5, 5, "north"
        self.score, self.energy = 0, 100
        self.items = {(5, 3): "blueLight"}
        # sala fechada 8x8 para o teste terminar rapido
        self.blocked = {(5, 1)}
        self.blocked |= {(8, y) for y in range(9)}
        self.blocked |= {(x, 8) for x in range(9)}
        self.pits = set()
        self.fell_in_pit = False
        self.reply_delay = 0.0   # simula latencia do servidor
        self.respawn = None      # segundos para itens reaparecerem
        self._respawn_at = {}
        self.actions = []

    def run(self):
        try:
            self._serve()
        except OSError:
            pass  # cliente desconectou no fim do teste

    def _serve(self):
        srv = socket.create_server(("127.0.0.1", self.port))
        conn, _ = srv.accept()
        buf = ""
        vec = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
        left = {"north": "west", "west": "south", "south": "east", "east": "north"}
        right = {v: k for k, v in left.items()}
        bumped = False
        while True:
            data = conn.recv(1024)
            if not data:
                break
            buf += data.decode()
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                cmd = line.strip().split(";")
                c = cmd[0]
                self.actions.append(c)
                if c == "w":
                    dx, dy = vec[self.dir]
                    nx, ny = self.x + dx, self.y + dy
                    if (nx, ny) in self.blocked or not (0 <= nx < 59 and 0 <= ny < 34):
                        bumped = True
                    else:
                        self.x, self.y = nx, ny
                        if (self.x, self.y) in self.pits:
                            self.fell_in_pit = True
                            self.score -= 1000
                elif c == "a":
                    self.dir = left[self.dir]
                elif c == "d":
                    self.dir = right[self.dir]
                elif c == "t":
                    if self.items.pop((self.x, self.y), None):
                        self.score += 1000
                        if self.respawn is not None:
                            self._respawn_at[(self.x, self.y)] = \
                                time.time() + self.respawn
                elif c == "o":
                    if self.reply_delay:
                        time.sleep(self.reply_delay)
                    for pos, when in list(self._respawn_at.items()):
                        if time.time() >= when:
                            self.items[pos] = "blueLight"
                            del self._respawn_at[pos]
                    obs = []
                    if bumped:
                        obs.append("blocked")
                        bumped = False
                    if self.items.get((self.x, self.y)) == "blueLight":
                        obs.append("blueLight")
                    adj = [(self.x, self.y - 1), (self.x + 1, self.y),
                           (self.x, self.y + 1), (self.x - 1, self.y)]
                    if any(p in self.pits for p in adj):
                        obs.append("breeze")
                    conn.sendall(f"o;{','.join(obs)}\n".encode())
                elif c == "q":
                    if self.reply_delay:
                        time.sleep(self.reply_delay)
                    conn.sendall(
                        f"s;{self.x};{self.y};{self.dir};game;{self.score};{self.energy}\n".encode())
                elif c == "g":
                    conn.sendall(b"g;Game;100000\n")
                elif c == "quit":
                    conn.close()
                    return


def test_smoke_agent():
    from communication import GameAI
    from strategy import DroneAgent

    random.seed(42)
    port = random.randint(20000, 30000)
    server = MiniServer(port)
    server.start()
    time.sleep(0.2)

    ai = GameAI()
    assert ai.connect("127.0.0.1", "Teste", port)
    ai.game_status = "Game"
    agent = DroneAgent(ai, log=lambda m: None)

    for _ in range(400):
        if server.score >= 1000:
            break
        agent.act()
        time.sleep(0.01)

    assert server.score >= 1000, f"agente nao pegou o tesouro (score={server.score})"
    assert "t" in server.actions and "w" in server.actions
    ai.client.disconnect()
    print(f"OK: smoke test (agente pegou tesouro, score simulado={server.score})")


def test_pit_avoidance():
    """Sala fechada com um poco: o agente deve explorar, pegar o tesouro
    e NUNCA cair no poco, mesmo passando perto dele."""
    from communication import GameAI
    from strategy import DroneAgent

    random.seed(7)
    port = random.randint(20000, 30000)
    server = MiniServer(port)
    server.pits = {(2, 5)}
    server.start()
    time.sleep(0.2)

    ai = GameAI()
    assert ai.connect("127.0.0.1", "TestePoco", port)
    ai.game_status = "Game"
    agent = DroneAgent(ai, log=lambda m: None)

    for _ in range(400):
        agent.act()
        assert not server.fell_in_pit, "agente caiu no poco!"
        if server.score >= 1000:
            break
        time.sleep(0.01)

    assert not server.fell_in_pit
    assert server.score >= 1000, f"nao pegou o tesouro (score={server.score})"
    ai.client.disconnect()
    print(f"OK: pit avoidance (explorou com poco no mapa, score={server.score})")


def test_stale_data_discarded():
    """Propriedade de seguranca: (1) sync retorna None quando a resposta
    nao chega no timeout (nunca entrega dados velhos); (2) um tick com
    sync None nao executa acao nem altera o modelo de mundo."""
    from communication import GameAI
    from strategy import DroneAgent
    from world_model import UNKNOWN

    random.seed(11)
    port = random.randint(20000, 30000)
    server = MiniServer(port)
    server.reply_delay = 3.0  # muito acima de qualquer timeout do agente
    server.start()
    time.sleep(0.2)

    ai = GameAI()
    assert ai.connect("127.0.0.1", "TesteLag", port)
    ai.game_status = "Game"

    # (1) timeout -> None, nunca dados velhos
    assert ai.request_sync_pair(timeout=0.3) is None

    # (2) tick descartado: nenhuma acao, mapa intacto
    agent = DroneAgent(ai, log=lambda m: None)
    agent.sync = lambda: None
    for _ in range(3):
        agent.act()
    moves = [a for a in server.actions if a in ("w", "s", "a", "d", "t", "e")]
    assert not moves, f"agente agiu com dados velhos: {moves}"
    assert all(cell == UNKNOWN for col in agent.world.grid for cell in col), \
        "mapa foi atualizado com dados velhos"
    ai.client.disconnect()
    print("OK: dados velhos descartados (timeout nao envenena o mapa)")


def test_planner_risk_and_flee_are_reachable():
    from planner import Planner
    from risk import RiskModel
    from world_model import DANGER_PIT, DANGER_FLASH

    w = WorldModel()
    risk = RiskModel(w)
    planner = Planner(w, risk)
    for x in range(5, 9):
        w.update_from_observation(x, 5, [])
    path = planner.a_star((5, 5), (8, 5), start_dir="east")
    assert path == [(6, 5), (7, 5), (8, 5)]
    w.grid[6][5] = DANGER_PIT
    w.version += 1
    assert planner.a_star((5, 5), (6, 5), start_dir="east") is None

    w2 = WorldModel()
    risk2 = RiskModel(w2)
    planner2 = Planner(w2, risk2)
    w2.update_from_observation(5, 5, [])
    w2.grid[6][5] = DANGER_FLASH
    w2.item_spots[(6, 5)] = "treasure"
    w2.version += 1
    assert planner2.a_star((5, 5), (6, 5), start_dir="east") is None
    assert planner2.a_star((5, 5), (6, 5), start_dir="east", allow_flash=True)

    w3 = WorldModel()
    risk3 = RiskModel(w3)
    planner3 = Planner(w3, risk3)
    for x in range(5, 8):
        w3.update_from_observation(x, 5, [])
    w3.update_from_observation(5, 6, [])
    w3.mark_blocked(6, 5)
    target, flee_path = planner3.flee_plan(
        (5, 5), "east", [(7, 5), (5, 6)],
        lambda c: risk3.flee_score(c, (5, 5)),
    )
    assert target == (5, 6), (target, flee_path)
    assert flee_path == [(5, 6)]
    old = planner3.a_star((5, 5), (5, 6), start_dir="south")
    w3.mark_blocked(5, 6)
    new = planner3.a_star((5, 5), (5, 6), start_dir="south")
    assert old and new is None
    print("OK: planner (BFS/A*, cache, teleporte fallback, fuga alcancavel)")


def test_risk_memory_does_not_poison_steps():
    from observations import Observation
    from risk import RiskModel

    w = WorldModel()
    risk = RiskModel(w)
    risk.update((5, 5), Observation.from_tokens(["steps"]))
    assert not w.danger_cells
    assert w.threat_until > time.time()
    risk.update((5, 5), Observation.from_tokens(["damage"]))
    assert (5, 5) in w.danger_cells
    assert risk.danger_penalty((5, 5)) > risk.danger_penalty((15, 15))
    print("OK: risk (steps nao vira perigo permanente; damage vira zona quente)")


def test_farming_utility_and_pending_grabs():
    from farm import FarmManager
    from planner import Planner
    from risk import RiskModel

    w = WorldModel()
    risk = RiskModel(w)
    planner = Planner(w, risk)
    farm = FarmManager(w, planner, risk, log=lambda m: None)
    for x in range(5, 12):
        w.update_from_observation(x, 5, [])
    w.item_spots[(6, 5)] = "powerup"
    w.item_spots[(11, 5)] = "treasure"
    plan = farm.choose_plan((5, 5), "east", energy=100)
    assert plan["goal"] == (11, 5), plan
    farm.record_grab((11, 5), 0, now=time.time())
    assert len(farm.pending_grabs) == 1
    farm.resolve_pending_grabs(1000, now=time.time() + 2)
    assert farm.item_values[(11, 5)] == 1000
    assert not farm.pending_grabs
    assert not farm.due_spots(("treasure",), 100)
    print("OK: farm (utilidade economica, pending_grabs, valor aprendido)")


def test_farming_avoids_recent_danger_zone():
    """Regressao/melhoria: a escolha de alvo de farm/exploracao quase nao
    levava risco recente em conta -- o desconto era FIXO (no maximo ~3.5 de
    uma ameaca recente de ate 70), irrelevante frente a farm_utility() (valor
    por segundo), que cresce muito rapido para alvos proximos: um alvo so 1-2
    passos mais perto ja vale centenas de pontos a mais, o que nenhum
    desconto fixo supera. Por isso o agente farmava de volta bem perto de
    onde acabou de ser atacado. O desconto agora e proporcional ao valor do
    proprio alvo (corta uma fracao dele), entao uma ameaca forte e recente
    consegue de fato virar a escolha mesmo quando o alvo perigoso e mais
    perto. Sobrevivencia importa em toda partida (o enunciado encerra a
    partida do agente ao morrer, nao so desconta -10 pontos)."""
    from farm import FarmManager
    from planner import Planner
    from risk import RiskModel
    from observations import Observation

    w = WorldModel()
    risk = RiskModel(w)
    planner = Planner(w, risk)
    farm = FarmManager(w, planner, risk, log=lambda m: None)

    start = (10, 10)
    spot_risky = (10, 7)   # 3 passos: mais perto, mas onde o agente acabou
                           # de levar tiro
    spot_safe = (10, 5)    # 5 passos: mais longe, sem ameaca recente

    w.mark_visited(*start)
    for y in range(4, 10):
        w.mark_safe(10, y)
    w.item_spots[spot_risky] = "treasure"
    w.item_spots[spot_safe] = "treasure"

    risk.update(spot_risky, Observation.from_tokens(["damage"]))

    plan = farm.choose_plan(start, "north", energy=100)
    assert plan is not None and plan["goal"] == spot_safe, (
        f"deveria preferir o alvo seguro {spot_safe} (mais longe) ao alvo "
        f"perigoso {spot_risky} (mais perto), escolheu {plan['goal'] if plan else None}")
    print("OK: farm evita zona de ameaca recente na escolha de alvo (nao so na fuga)")


def test_combat_rules():
    from combat import CombatController, FIRST_ENGAGEMENT_MISS_LIMIT
    from observations import Observation
    from risk import RiskModel

    w = WorldModel()
    risk = RiskModel(w)
    combat = CombatController(w, risk, log=lambda m: None)
    for y in range(2, 6):
        w.update_from_observation(5, y, [])
    w.update_pose(5, 5, "north")
    assert not combat.should_attack(100, Observation.from_tokens(["steps"]))
    assert combat.should_attack(100, Observation.from_tokens(["enemy#3"]))
    w.mark_blocked(5, 4)
    assert not combat.should_attack(100, Observation.from_tokens(["enemy#3"]))
    assert combat.decide(100, Observation.from_tokens(["enemy#8"])) in (None, "CHASE")
    assert combat.decide(20, Observation.from_tokens(["enemy#3"])) == "EVADE"
    # energia baixa: sempre evade ao levar dano, mesmo com agressividade alta
    assert combat.decide(20, Observation.from_tokens(["damage"])) == "EVADE"
    # energia/agressividade suficientes: contra-ataca em vez de so fugir
    assert combat.decide(60, Observation.from_tokens(["damage"])) == "HUNT"
    assert combat.hunt_reason == "damage"
    combat.record_shot()
    combat.update_after_observation(Observation.from_tokens(["hit"]))
    assert combat.shots_hit == 1 and combat.shots_since_hit == 0
    combat.shots_since_hit = combat.miss_limit(6)
    assert combat.after_miss_limit(6)

    # varios tiros seguidos sem NENHUM acerto: desengaja e pausa o combate
    unlucky = CombatController(w, risk, log=lambda m: None)
    unlucky.shots_fired = FIRST_ENGAGEMENT_MISS_LIMIT
    unlucky.shots_hit = 0
    w.grid[5][4] = VISITED
    assert not unlucky.should_attack(100, Observation.from_tokens(["enemy#3"]))
    assert unlucky.engage_pause_until > time.time()
    print("OK: combat (linha de tiro, steps, hit, miss limit, evade/chase)")


class FakeAI:
    def __init__(self, views):
        self.views = list(views)
        self.actions = []
        self.score = 0
        self.energy = 100
        self.x = 5
        self.y = 5
        self.direction = "north"
        self.observation = []
        self.pending_events = []

    def request_sync_pair(self, timeout=0.5):
        self.actions.extend(["q", "o"])
        if not self.views:
            return (self.x, self.y, self.direction, "game", self.score, self.energy,
                    list(self.observation))
        view = self.views.pop(0)
        self.x, self.y, self.direction = view[0], view[1], view[2]
        self.score = view[4]
        self.energy = view[5]
        self.observation = list(view[6])
        return view

    def request_status_sync(self, timeout=0.5):
        self.actions.append("q")
        return self.x, self.y, self.direction, "game", self.score, self.energy

    def request_observation_sync(self, timeout=0.5):
        self.actions.append("o")
        obs = list(self.observation)
        obs.extend(self.consume_pending_events())
        return obs

    def consume_pending_events(self):
        events = list(self.pending_events)
        self.pending_events.clear()
        return events

    def send_forward(self):
        self.actions.append("w")
        vec = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
        dx, dy = vec[self.direction]
        self.x += dx
        self.y += dy
    def send_backward(self): self.actions.append("s")
    def send_turn_left(self):
        self.actions.append("a")
        self.direction = {"north": "west", "west": "south", "south": "east", "east": "north"}[self.direction]
    def send_turn_right(self):
        self.actions.append("d")
        self.direction = {"north": "east", "east": "south", "south": "west", "west": "north"}[self.direction]
    def send_get_item(self):
        self.actions.append("t")
        self.observation = []
    def send_shoot(self): self.actions.append("e")


def test_strategy_fsm_and_reset():
    from strategy import DroneAgent

    ai = FakeAI([(5, 5, "north", "game", 0, 60, ["blueLight"])])
    agent = DroneAgent(ai, log=lambda m: None)
    agent.act()
    assert "t" in ai.actions and agent.state == "EXPLORE"

    ai2 = FakeAI([(5, 5, "north", "game", 0, 20, ["damage"])])
    agent2 = DroneAgent(ai2, log=lambda m: None)
    agent2.world.update_from_observation(5, 5, [])
    agent2.world.update_from_observation(6, 5, [])
    agent2.act()
    assert agent2.state in ("EVADE", "FLEE", "EXPLORE")

    agent3 = DroneAgent(FakeAI([]), log=lambda m: None)
    agent3.world.update_from_observation(1, 1, [])
    assert agent3.world.visit_count
    agent3.reset_for_new_game()
    assert not agent3.world.visit_count and agent3.state == "EXPLORE"
    print("OK: strategy (FSM e reset completo)")


def test_economical_observation_policy():
    from strategy import DroneAgent

    ai = FakeAI([(5, 5, "north", "game", 0, 100, [])])
    agent = DroneAgent(ai, log=lambda m: None)
    agent.act()
    ai.actions.clear()

    agent.world.item_spots[(5, 5)] = "treasure"
    agent.farm.last_taken[(5, 5)] = time.time()
    agent.farm.spot_respawn[(5, 5)] = 30.0
    agent.farm.spot_next_check[(5, 5)] = time.time() + 30.0
    agent.state = "CAMP"
    agent.farm.has_better_reachable_plan = lambda *args, **kwargs: False
    agent._need_status = False
    agent._need_observation = False
    agent._last_status_at = time.time()
    agent._last_observation_at = time.time()

    for _ in range(20):
        agent.act()
    assert "o" not in ai.actions and "q" not in ai.actions, ai.actions
    assert agent.metrics.idle_without_command >= 20
    print("OK: economia de observacao (camp espera sem q/o continuo)")


def test_observe_after_movement_not_each_idle_tick():
    from strategy import DroneAgent

    ai = FakeAI([])
    agent = DroneAgent(ai, log=lambda m: None)
    agent._cached_view = (5, 5, "north", "game", 0, 100, [])
    agent._last_status_at = time.time()
    agent._last_observation_at = time.time()
    agent._need_status = False
    agent._need_observation = False
    ai.x, ai.y, ai.direction = 5, 5, "north"
    agent.world.update_from_observation(5, 5, [])
    agent.world.update_from_observation(5, 4, [])
    agent.path = [(5, 4)]
    agent.act()
    assert "w" in ai.actions
    agent.act()
    assert "q" in ai.actions and "o" in ai.actions, ai.actions
    print("OK: economia de observacao (movimento solicita q/o no tick seguinte)")


def test_steps_do_not_cancel_safe_route_or_shoot():
    from strategy import DroneAgent

    ai = FakeAI([])
    agent = DroneAgent(ai, log=lambda m: None)
    agent._cached_view = (5, 5, "north", "game", 0, 100, ["steps"])
    agent._last_status_at = time.time()
    agent._last_observation_at = time.time()
    agent._need_status = False
    agent._need_observation = False
    ai.x, ai.y, ai.direction = 5, 5, "north"
    ai.observation = ["steps"]
    agent.world.update_from_observation(5, 5, [])
    agent.world.update_from_observation(5, 4, [])
    agent.path = [(5, 4)]
    agent.act()
    assert "e" not in ai.actions
    assert "w" in ai.actions
    print("OK: steps isolado nao atira nem cancela rota segura")


def test_camp_without_current_light_is_deferred():
    from strategy import DroneAgent

    ai = FakeAI([])
    agent = DroneAgent(ai, log=lambda m: None)
    agent._cached_view = (6, 12, "north", "game", 0, 100, [])
    agent._last_status_at = time.time()
    agent._last_observation_at = time.time()
    agent._need_status = False
    agent._need_observation = False
    ai.x, ai.y, ai.direction = 6, 12, "north"
    agent.world.update_from_observation(6, 12, [])
    agent.world.item_spots[(6, 12)] = "treasure"
    agent.state = "CAMP"
    agent.act()
    assert agent.state != "CAMP"
    assert agent.farm.spot_next_check[(6, 12)] > time.time()
    ai.actions.clear()
    for _ in range(5):
        agent.act()
    assert agent.state != "CAMP"
    assert any(a in ai.actions for a in ("w", "a", "d")) or agent.metrics.idle_without_command > 0
    print("OK: camp sem luz atual e adiado para evitar loop estatico")


def prepare_blocked_chase_agent(obs=None):
    from strategy import DroneAgent

    ai = FakeAI([])
    agent = DroneAgent(ai, log=lambda m: None)
    tokens = obs if obs is not None else ["enemy#6"]
    agent._cached_view = (26, 23, "east", "game", 0, 100, tokens)
    agent._last_status_at = time.time()
    agent._last_observation_at = time.time()
    agent._need_status = False
    agent._need_observation = False
    ai.x, ai.y, ai.direction = 26, 23, "east"
    ai.observation = list(tokens)
    agent.world.update_from_observation(26, 23, [])
    agent.world.update_from_observation(26, 22, [])
    agent.world.update_from_observation(26, 24, [])
    agent.world.update_from_observation(25, 23, [])
    agent.world.mark_blocked(27, 23)
    agent.world.update_pose(26, 23, "east")
    return agent, ai


def test_blocked_chase_does_not_oscillate_hunt_chase():
    agent, ai = prepare_blocked_chase_agent()
    for _ in range(8):
        agent.act()
    snap = agent.metrics_snapshot()
    assert snap["transitions"].get("CHASE->HUNT", 0) <= 1, snap
    assert snap["transitions"].get("HUNT->CHASE", 0) == 0, snap
    assert agent.state in ("REPOSITION", "EVADE", "EXPLORE", "ATTACK", "FLEE", "CHASE")
    assert snap["blocked_chase_failures"] >= 1
    assert "w" in ai.actions or "d" in ai.actions or snap["chases_abandoned"] >= 1
    print("OK: chase bloqueado nao oscila CHASE/HUNT")


def test_repeated_blocked_chase_adds_cooldown_and_changes_strategy():
    agent, ai = prepare_blocked_chase_agent()
    agent.act()
    first = agent.metrics_snapshot()["blocked_chase_failures"]
    agent._set_state("CHASE", "teste repeticao")
    agent.act()
    snap = agent.metrics_snapshot()
    assert snap["blocked_chase_failures"] == first
    assert snap["repeated_blocks"] >= 1
    assert agent.state != "CHASE"
    assert ai.actions.count("w") <= 1
    print("OK: mesmo bloqueio ativa cooldown e evita repetir chase")


def test_stale_enemy_observation_does_not_reactivate_chase():
    agent, ai = prepare_blocked_chase_agent()
    agent._enemy_seen_at = time.time() - 10
    agent.act()
    assert agent.state != "CHASE"
    assert agent.metrics_snapshot()["chases_abandoned"] >= 1 or "e" not in ai.actions
    print("OK: enemy antigo expira e nao reativa chase")


def test_blocked_chase_repositions_laterally_when_safe():
    agent, ai = prepare_blocked_chase_agent()
    agent.act()
    assert agent.state in ("REPOSITION", "HUNT", "EXPLORE", "ATTACK")
    for _ in range(3):
        agent.act()
    assert "w" in ai.actions or agent.metrics_snapshot()["repositions"] >= 1
    assert (agent.world.position != (26, 23)) or "w" in ai.actions
    print("OK: chase bloqueado tenta reposicionamento lateral seguro")


def test_blocked_chase_without_flank_evades_or_abandons():
    agent, ai = prepare_blocked_chase_agent()
    # bloqueia as 3 direcoes alcancaveis (frente ja esta bloqueada pelo
    # setup): sem flanco em lugar nenhum, so resta abandonar a perseguicao.
    agent.world.mark_blocked(26, 22)
    agent.world.mark_blocked(26, 24)
    agent.world.mark_blocked(25, 23)
    agent.act()
    for _ in range(3):
        agent.act()
    assert agent.state in ("EVADE", "FLEE", "EXPLORE")
    assert agent.metrics_snapshot()["transitions"].get("HUNT->CHASE", 0) == 0
    print("OK: sem flanqueamento, abandona/foge em vez de hunt infinito")


def test_failed_chase_restores_previous_farm_plan():
    import strategy

    agent, ai = prepare_blocked_chase_agent()
    agent.world.update_from_observation(25, 23, [])
    agent._previous_plan = {
        "goal": (25, 23),
        "path": [(25, 23)],
        "allows_flash": False,
        "saved_at": time.time(),
    }
    original_limit = strategy.MAX_BLOCKED_CHASE_FAILURES
    strategy.MAX_BLOCKED_CHASE_FAILURES = 0
    try:
        agent.act()
    finally:
        strategy.MAX_BLOCKED_CHASE_FAILURES = original_limit
    assert agent.goal == (25, 23) or agent.metrics_snapshot()["previous_plan_restored"] >= 1
    assert agent.state == "EXPLORE"
    print("OK: chase falho pode restaurar plano de farm/exploracao anterior")


def test_farming():
    """Itens reaparecem: numa sala minuscula ja explorada, o agente deve
    ACAMPAR sobre o ponto de tesouro e coleta-lo a cada respawn,
    multiplicando a pontuacao."""
    from communication import GameAI
    from strategy import DroneAgent

    random.seed(5)
    port = random.randint(20000, 30000)
    server = MiniServer(port)
    # sala 3x3 (x 4..6, y 3..5) com tesouro em (5,4); respawn rapido
    server.blocked = {(3, yy) for yy in range(2, 7)} | \
                     {(7, yy) for yy in range(2, 7)} | \
                     {(xx, 2) for xx in range(3, 8)} | \
                     {(xx, 6) for xx in range(3, 8)}
    server.items = {(5, 4): "blueLight"}
    server.respawn = 0.3
    server.start()
    time.sleep(0.2)

    ai = GameAI()
    assert ai.connect("127.0.0.1", "TesteFarm", port)
    ai.game_status = "Game"
    agent = DroneAgent(ai, log=lambda m: None)

    deadline = time.time() + 8
    while time.time() < deadline and server.score < 3000:
        agent.act()
        time.sleep(0.01)

    assert server.score >= 3000, \
        f"farming insuficiente (score={server.score}, esperado >= 3000)"
    ai.client.disconnect()
    print(f"OK: farming (coletas repetidas no mesmo ponto, score={server.score})")


def test_damage_reaction_hunts_when_strong_evades_when_weak():
    """Ao levar tiro, o agente contra-ataca (HUNT) quando tem energia e
    agressividade suficientes, e evade quando esta fraco. decide_state()
    delega essa escolha inteira a combat.decide() -- nao ha atalho que
    sempre force fuga, como havia antes dessa reacao ser corrigida."""
    from combat import CombatController
    from observations import Observation
    from risk import RiskModel
    from strategy import DroneAgent

    w = WorldModel()
    risk = RiskModel(w)
    combat = CombatController(w, risk, log=lambda m: None)
    dmg = Observation.from_tokens(["damage"])

    assert combat.decide(60, dmg) == "HUNT" and combat.hunt_reason == "damage"
    assert combat.decide(20, dmg) == "EVADE"  # energia baixa: sempre evade

    ai = FakeAI([(5, 5, "north", "game", 0, 60, ["damage"])])
    agent = DroneAgent(ai, log=lambda m: None)
    agent.act()
    assert agent.state == "HUNT", f"esperado HUNT, obtido {agent.state}"

    ai_low = FakeAI([(5, 5, "north", "game", 0, 15, ["damage"])])
    agent_low = DroneAgent(ai_low, log=lambda m: None)
    agent_low.act()
    assert agent_low.state == "EVADE", f"energia critica deveria evadir, obtido {agent_low.state}"

    print("OK: reacao a damage (HUNT quando forte, EVADE quando fraco/energia baixa)")


def test_survey_state_removed():
    """SURVEY era um estado morto: existia em STATES, tinha handler e
    dispatch, mas nada em strategy.py jamais setava state='SURVEY' nem
    survey_turns>0 (achado da auditoria). Foi removido em vez de receber
    uma condicao de entrada nova. Este teste garante que ele nao volta a
    existir 'a toa' (nome sem comportamento) na FSM."""
    from strategy import DroneAgent

    ai = FakeAI([])
    agent = DroneAgent(ai, log=lambda m: None)
    assert "SURVEY" not in agent.STATES
    assert not hasattr(agent, "survey_turns")
    assert not hasattr(agent, "do_survey")
    print("OK: SURVEY removido (sem estado morto na FSM)")


def test_no_stale_observation_reuse_after_critical_actions():
    """Regressao (classe de bug ja encontrada uma vez em do_grab nesta
    sessao): qualquer acao que muda o estado do lado do servidor precisa
    forcar uma observacao fresca no proximo tick, em vez de reaproveitar a
    observacao em cache de ANTES da acao. Cobre GRAB, SHOOT, 'blocked' e
    mudanca de celula."""
    from strategy import DroneAgent

    # 1) apos GRAB: observacao fresca exigida e luz obsoleta removida do cache
    ai = FakeAI([(5, 5, "north", "game", 0, 100, ["blueLight"])])
    agent = DroneAgent(ai, log=lambda m: None)
    agent.act()
    assert agent.last_action == "grab"
    assert agent._need_observation is True, "GRAB deve forcar reobservacao no proximo tick"
    _, _, _, _, _, _, tokens = agent._cached_view
    assert "bluelight" not in [t.lower() for t in tokens], \
        "luz obsoleta deveria ter sido limpa do cache apos o grab"

    # 2) apos SHOOT: observacao fresca exigida para saber se houve 'hit'
    ai2 = FakeAI([(5, 5, "north", "game", 0, 100, ["enemy#2"])])
    agent2 = DroneAgent(ai2, log=lambda m: None)
    agent2.world.update_from_observation(5, 5, [])
    agent2.act()
    assert agent2.last_action == "shoot", f"esperado tiro, obtido {agent2.last_action}"
    assert agent2._need_observation is True, "tiro deve forcar reobservacao (hit/miss)"

    # 3) apos 'blocked': mapa/rota precisam ser recalculados com dado fresco
    ai3 = FakeAI([(5, 5, "north", "game", 0, 100, ["blocked"])])
    agent3 = DroneAgent(ai3, log=lambda m: None)
    agent3.path = [(5, 4)]
    agent3.act()
    assert agent3._need_observation is True, "'blocked' deve forcar reobservacao"

    # 4) apos mudanca de celula (forward bem-sucedido): o tick seguinte nao
    # deve decidir com o status/observacao antigos da celula anterior
    ai4 = FakeAI([])
    agent4 = DroneAgent(ai4, log=lambda m: None)
    agent4._cached_view = (5, 5, "north", "game", 0, 100, [])
    agent4._last_status_at = time.time()
    agent4._last_observation_at = time.time()
    agent4._need_status = False
    agent4._need_observation = False
    ai4.x, ai4.y, ai4.direction = 5, 5, "north"
    agent4.world.update_from_observation(5, 5, [])
    agent4.world.update_from_observation(5, 4, [])
    agent4.path = [(5, 4)]
    agent4.act()
    assert "w" in ai4.actions
    assert agent4._need_status is True and agent4._need_observation is True, \
        "apos mudar de celula, o proximo tick precisa pedir status/observacao novos"

    print("OK: sem reuso de observacao obsoleta apos GRAB/SHOOT/blocked/mudanca de celula")


def test_player_state_reset_on_new_game():
    """Regressao: ai.player_state so e atualizado quando chega uma mensagem
    's' do servidor, recebida dentro de agent.act(). Se o drone morreu na
    partida anterior (player_state='dead') e uma partida nova comeca, o loop
    principal nunca chama agent.act() enquanto acha que ainda esta morto --
    travando para sempre, mesmo com o drone vivo de novo. main.py agora
    reseta player_state ao detectar uma partida nova; sem esse reset, o loop
    (rodado aqui numa thread com timeout) nunca terminaria."""
    import main as main_module

    class FakeGameAI:
        def __init__(self, log=print):
            self.connected = True
            self.game_status = "Game"
            self.player_state = "dead"  # sobra da partida anterior
            self.score = 0
            self.energy = 100
            self.player_x = 5
            self.player_y = 5
            self.player_dir = "north"

        def connect(self, host, name, port=8888):
            return True

        def send_color(self, r, g, b):
            pass

        def send_request_game_status(self):
            pass

        def request_scoreboard_sync(self, timeout=0.4):
            return []

        def disconnect(self):
            self.connected = False

    class FakeFarm:
        collect_by_kind = {}
        collect_count = 0

    class FakeWorld:
        item_spots = {}

    created_agents = []

    class FakeAgent:
        def __init__(self, ai, log=print):
            self.ai = ai
            self.state = "EXPLORE"
            self.config = {}
            self.farm = FakeFarm()
            self.world = FakeWorld()
            self.acted = 0
            created_agents.append(self)

        def act(self):
            self.acted += 1
            self.ai.connected = False  # encerra o loop apos 1 acao bem-sucedida

        def metrics_snapshot(self):
            return {}

    fake_ai = FakeGameAI()
    orig_game_ai, orig_agent = main_module.GameAI, main_module.DroneAgent
    main_module.GameAI = lambda log=print: fake_ai
    main_module.DroneAgent = FakeAgent
    try:
        thread = threading.Thread(target=main_module.main, args=(["127.0.0.1", "Teste"],), daemon=True)
        thread.start()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), (
            "loop principal travou: player_state 'dead' da partida anterior "
            "nunca foi resetado, entao agent.act() nunca roda na partida nova")
        # main() cria um DroneAgent antes do loop e outro ao detectar a
        # partida nova (was_in_game False -> True); o que importa e que
        # ALGUM deles chegou a agir (nao travou antes disso).
        assert created_agents and any(a.acted >= 1 for a in created_agents)
    finally:
        main_module.GameAI, main_module.DroneAgent = orig_game_ai, orig_agent
    print("OK: player_state 'dead' da partida anterior nao trava a partida nova")


if __name__ == "__main__":
    test_observation_normalization()
    test_async_notifications_are_preserved()
    test_world_model()
    test_world_model_extended_features()
    test_fuzzy()
    test_planner_risk_and_flee_are_reachable()
    test_risk_memory_does_not_poison_steps()
    test_farming_utility_and_pending_grabs()
    test_farming_avoids_recent_danger_zone()
    test_combat_rules()
    test_damage_reaction_hunts_when_strong_evades_when_weak()
    test_survey_state_removed()
    test_no_stale_observation_reuse_after_critical_actions()
    test_strategy_fsm_and_reset()
    test_player_state_reset_on_new_game()
    test_economical_observation_policy()
    test_observe_after_movement_not_each_idle_tick()
    test_steps_do_not_cancel_safe_route_or_shoot()
    test_camp_without_current_light_is_deferred()
    test_blocked_chase_does_not_oscillate_hunt_chase()
    test_repeated_blocked_chase_adds_cooldown_and_changes_strategy()
    test_stale_enemy_observation_does_not_reactivate_chase()
    test_blocked_chase_repositions_laterally_when_safe()
    test_blocked_chase_without_flank_evades_or_abandons()
    test_failed_chase_restores_previous_farm_plan()
    test_smoke_agent()
    test_pit_avoidance()
    test_stale_data_discarded()
    test_farming()
    print("Todos os testes passaram.")
