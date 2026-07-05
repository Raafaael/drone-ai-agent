"""
DevKit em Python para o INF1771 - Desafio dos Drones.

Implementa a comunicacao via Socket TCP/IP (porta 8888) com o GameServer,
seguindo o protocolo texto descrito no enunciado (parametros separados
por ';' e observacoes separadas por ',').
"""

import socket
import threading
import time


class HandleClient:
    """Camada baixa: socket TCP + thread de recepcao de mensagens."""

    def __init__(self):
        self.sock = None
        self.connected = False
        self._recv_thread = None
        self._buffer = ""
        self.msg_handlers = []  # callbacks: f(cmd_list)

    def connect(self, host, port=8888):
        try:
            self.sock = socket.create_connection((host, port), timeout=10)
            self.sock.settimeout(None)
            self.connected = True
            self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._recv_thread.start()
            return True
        except OSError as e:
            print(f"[REDE] Falha ao conectar em {host}:{port} -> {e}")
            self.connected = False
            return False

    def disconnect(self):
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def send_msg(self, msg):
        if not self.connected or not self.sock:
            return False
        try:
            self.sock.sendall((msg + "\n").encode("utf-8"))
            return True
        except OSError as e:
            print(f"[REDE] Erro ao enviar '{msg}': {e}")
            self.connected = False
            return False

    def _receive_loop(self):
        while self.connected:
            try:
                data = self.sock.recv(4096)
            except OSError:
                break
            if not data:
                break
            self._buffer += data.decode("utf-8", errors="ignore")
            # mensagens terminadas em quebra de linha
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.strip("\r ")
                if not line:
                    continue
                cmd = line.split(";")
                for handler in self.msg_handlers:
                    try:
                        handler(cmd)
                    except Exception as e:  # nao derruba a thread de rede
                        print(f"[REDE] Erro no handler: {e}")
        self.connected = False
        print("[REDE] Conexao encerrada.")


class GameAI:
    """
    Camada do protocolo do jogo: comandos de controle do drone e
    parsing das mensagens do servidor. Mantem o ultimo estado conhecido
    (posicao, direcao, energia, pontos, estado do jogo) e a lista de
    observacoes mais recente.
    """

    DIRECTIONS = ("north", "east", "south", "west")

    def __init__(self):
        self.client = HandleClient()
        self.client.msg_handlers.append(self._handle_message)

        self.player_x = -1
        self.player_y = -1
        self.player_dir = "north"
        self.player_state = "ready"   # ready | game | dead | gameover
        self.score = 0
        self.energy = 100

        self.game_status = "Ready"
        self.game_time = 0

        self.scoreboard = []
        self.observations = []        # ultima lista de observacoes recebida
        self.pending_observations = []  # hit/damage recebidos fora do comando 'o'
        self.obs_event = threading.Event()
        self.status_event = threading.Event()
        self.lock = threading.Lock()

    # ---------------- conexao ----------------

    def connect(self, host, name, port=8888):
        if not self.client.connect(host, port):
            return False
        time.sleep(0.2)
        self.send_name(name)
        return True

    def disconnect(self):
        self.send_goodbye()
        self.client.disconnect()

    @property
    def connected(self):
        return self.client.connected

    # ---------------- comandos (enunciado) ----------------

    def send_forward(self):           self.client.send_msg("w")
    def send_backward(self):          self.client.send_msg("s")
    def send_turn_left(self):         self.client.send_msg("a")
    def send_turn_right(self):        self.client.send_msg("d")
    def send_get_item(self):          self.client.send_msg("t")
    def send_shoot(self):             self.client.send_msg("e")
    def send_request_observation(self): self.client.send_msg("o")
    def send_request_game_status(self): self.client.send_msg("g")
    def send_request_user_status(self): self.client.send_msg("q")
    def send_request_position(self):  self.client.send_msg("p")
    def send_request_scoreboard(self): self.client.send_msg("u")
    def send_goodbye(self):           self.client.send_msg("quit")
    def send_name(self, name):        self.client.send_msg(f"name;{name}")
    def send_say(self, msg):          self.client.send_msg(f"say;{msg}")
    def send_color(self, r, g, b):    self.client.send_msg(f"color;{r};{g};{b}")

    # ---------------- parsing das mensagens ----------------

    def _remember_async_observation(self, obs_name):
        """Guarda notificacoes avulsas (hit/damage) para o proximo tick.

        Alguns servidores enviam acerto/dano fora da resposta do comando
        'o'. Se isso for sobrescrito pela proxima observacao, o agente atira
        e desvia como se nada tivesse acontecido.
        """
        with self.lock:
            if obs_name not in self.pending_observations:
                self.pending_observations.append(obs_name)
            if obs_name not in self.observations:
                self.observations.append(obs_name)

    def _handle_message(self, cmd):
        head = cmd[0].lower()

        if head == "o":  # observacoes
            obs = []
            if len(cmd) > 1 and cmd[1] != "":
                obs = [o.strip() for o in cmd[1].split(",") if o.strip()]
            with self.lock:
                if self.pending_observations:
                    seen = {o.lower() for o in obs}
                    for pending in self.pending_observations:
                        if pending.lower() not in seen:
                            obs.append(pending)
                    self.pending_observations.clear()
                self.observations = obs
            self.obs_event.set()

        elif head == "s":  # status do usuario: x;y;dir;state;score;energy
            try:
                with self.lock:
                    self.player_x = int(cmd[1])
                    self.player_y = int(cmd[2])
                    self.player_dir = cmd[3].lower()
                    self.player_state = cmd[4].lower()
                    self.score = int(cmd[5])
                    self.energy = int(cmd[6])
                self.status_event.set()
            except (IndexError, ValueError):
                pass

        elif head == "p":  # posicao: x;y[;dir]
            try:
                with self.lock:
                    self.player_x = int(cmd[1])
                    self.player_y = int(cmd[2])
                    if len(cmd) > 3:
                        self.player_dir = cmd[3].lower()
            except (IndexError, ValueError):
                pass

        elif head == "g":  # status do jogo: estado;tempo
            try:
                with self.lock:
                    self.game_status = cmd[1]
                    if len(cmd) > 2:
                        self.game_time = int(cmd[2])
            except (IndexError, ValueError):
                pass

        elif head == "u":  # scoreboard
            with self.lock:
                self.scoreboard = cmd[1:]

        elif head == "notification":
            for part in cmd[1:]:
                token = part.strip().lower()
                if token in ("hit", "damage"):
                    self._remember_async_observation(token)
            print(f"[SERVIDOR] {';'.join(cmd[1:])}")

        elif head == "hello":
            print(f"[SERVIDOR] Jogador conectado: {';'.join(cmd[1:])}")

        elif head == "goodbye":
            print(f"[SERVIDOR] Jogador saiu: {';'.join(cmd[1:])}")

        elif head == "changename":
            pass

        elif head in ("h", "d"):  # hit / damage avulsos (alguns servidores)
            self._remember_async_observation("hit" if head == "h" else "damage")

    # ---------------- utilidades sincronas ----------------

    def request_observation_sync(self, timeout=1.0):
        """Pede observacoes e espera a resposta. Retorna lista (pode ser vazia)."""
        self.obs_event.clear()
        self.send_request_observation()
        self.obs_event.wait(timeout)
        with self.lock:
            return list(self.observations)

    def request_status_sync(self, timeout=1.0):
        """Pede status do usuario e espera. Retorna (x, y, dir, state, score, energy)."""
        self.status_event.clear()
        self.send_request_user_status()
        self.status_event.wait(timeout)
        with self.lock:
            return (self.player_x, self.player_y, self.player_dir,
                    self.player_state, self.score, self.energy)

    def request_sync_pair(self, timeout=0.5):
        """Pede status + observacoes em paralelo (1 ida-e-volta em vez de 2).
        Retorna (x, y, dir, state, score, energy, observacoes) ou None se
        alguma resposta nao chegou a tempo.

        IMPORTANTE: nunca retorna dados velhos. Usar posicao/observacao de
        ticks anteriores atribuiria sensores a celula errada e envenenaria
        o modelo de mundo (ex.: marcar como seguro o vizinho de um poco)."""
        self.status_event.clear()
        self.obs_event.clear()
        self.send_request_user_status()
        self.send_request_observation()
        ok = self.status_event.wait(timeout) and self.obs_event.wait(timeout)
        if not ok:
            return None
        with self.lock:
            return (self.player_x, self.player_y, self.player_dir,
                    self.player_state, self.score, self.energy,
                    list(self.observations))
