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


class DummyAI:
    def __init__(self):
        self.actions = []

    def send_shoot(self): self.actions.append("e")
    def send_turn_right(self): self.actions.append("d")
    def send_get_item(self): self.actions.append("t")
    def send_forward(self): self.actions.append("w")
    def send_turn_left(self): self.actions.append("a")


def test_combat_fire_control():
    """Tiro custa -10: inimigo longe ou baixa taxa de acerto nao deve
    sequestrar a estrategia principal do agente."""
    from ai_agent import DroneAgent

    agent = DroneAgent(DummyAI(), log=lambda m: None)
    assert agent.decide_state(0, 0, 50, ["enemy#8"]) == "EXPLORE"
    assert agent.decide_state(0, 0, 90, ["enemy#4"]) == "ATTACK"

    agent.shots_fired = 8
    agent.shots_hit = 0
    assert agent.decide_state(0, 0, 90, ["enemy#5"]) == "EXPLORE"
    print("OK: combat fire control (economia de tiros)")


def test_scored_planning_avoids_local_loop():
    """Entre dois alvos alcancaveis, o planejador deve preferir sair da
    regiao hiper-revisitada em vez de orbitar o vizinho mais proximo."""
    from ai_agent import DroneAgent

    agent = DroneAgent(DummyAI(), log=lambda m: None)
    for x in range(5, 10):
        agent.world.update_from_observation(x, 5, [])
        agent.seen_cells.add((x, 5))

    agent.recent_positions = [(6, 5), (5, 5)] * 13
    agent.world.visit_count[(6, 5)] = 14
    assert agent._plan_to_scored_targets(5, 5, "east", [(6, 5), (9, 5)],
                                         "PLANO", 100)
    assert agent.goal == (9, 5), agent.goal
    print("OK: scored planning (anti-loop local)")


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
    from devkit import GameAI
    from ai_agent import DroneAgent

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
    from devkit import GameAI
    from ai_agent import DroneAgent

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
    from devkit import GameAI
    from ai_agent import DroneAgent
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


def test_farming():
    """Itens reaparecem: numa sala minuscula ja explorada, o agente deve
    ACAMPAR sobre o ponto de tesouro e coleta-lo a cada respawn,
    multiplicando a pontuacao."""
    from devkit import GameAI
    from ai_agent import DroneAgent

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


if __name__ == "__main__":
    test_world_model()
    test_fuzzy()
    test_combat_fire_control()
    test_scored_planning_avoids_local_loop()
    test_smoke_agent()
    test_pit_avoidance()
    test_stale_data_discarded()
    test_farming()
    print("Todos os testes passaram.")
