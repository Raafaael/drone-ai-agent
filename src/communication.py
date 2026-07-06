"""TCP/IP protocol client for the INF1771 drone GameServer."""

import socket
import threading
import time

from observations import Observation, normalize_token, normalize_tokens, parse_observation_payload


class HandleClient:
    """Low-level TCP client with continuous receive thread."""

    def __init__(self, log=print):
        self.sock = None
        self.connected = False
        self._recv_thread = None
        self._buffer = ""
        self._lock = threading.Lock()
        self.msg_handlers = []
        self.log = log

    def connect(self, host, port=8888):
        try:
            self.sock = socket.create_connection((host, port), timeout=10)
            self.sock.settimeout(0.5)
            self.connected = True
            self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._recv_thread.start()
            return True
        except OSError as exc:
            self.log(f"[REDE] Falha ao conectar em {host}:{port} -> {exc}")
            self.connected = False
            return False

    def disconnect(self):
        self.connected = False
        sock = self.sock
        self.sock = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        thread = self._recv_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)

    def send_msg(self, msg):
        with self._lock:
            if not self.connected or not self.sock:
                return False
            try:
                self.sock.sendall((msg + "\n").encode("utf-8"))
                return True
            except OSError as exc:
                self.log(f"[REDE] Erro ao enviar '{msg}': {exc}")
                self.connected = False
                return False

    def _receive_loop(self):
        while self.connected:
            try:
                data = self.sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            self._buffer += data.decode("utf-8", errors="ignore")
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.strip("\r ")
                if not line:
                    continue
                cmd = line.split(";")
                for handler in list(self.msg_handlers):
                    try:
                        handler(cmd)
                    except Exception as exc:
                        self.log(f"[REDE] Erro no handler: {exc}")
        self.connected = False
        self.log("[REDE] Conexao encerrada.")


class GameAI:
    """Thread-safe GameServer protocol facade."""

    DIRECTIONS = ("north", "east", "south", "west")

    def __init__(self, log=print):
        self.client = HandleClient(log=log)
        self.client.msg_handlers.append(self._handle_message)
        self.log = log

        self.player_x = -1
        self.player_y = -1
        self.player_dir = "north"
        self.player_state = "ready"
        self.score = 0
        self.energy = 100
        self.game_status = "Ready"
        self.game_time = 0
        self.scoreboard = []
        self.observations = []
        self.latest_observation = Observation()
        self.pending_observations = []

        self.obs_event = threading.Event()
        self.status_event = threading.Event()
        self.game_event = threading.Event()
        self.scoreboard_event = threading.Event()
        self.lock = threading.RLock()

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

    def send_forward(self): return self.client.send_msg("w")
    def send_backward(self): return self.client.send_msg("s")
    def send_turn_left(self): return self.client.send_msg("a")
    def send_turn_right(self): return self.client.send_msg("d")
    def send_get_item(self): return self.client.send_msg("t")
    def send_shoot(self): return self.client.send_msg("e")
    def send_request_observation(self): return self.client.send_msg("o")
    def send_request_game_status(self): return self.client.send_msg("g")
    def send_request_user_status(self): return self.client.send_msg("q")
    def send_request_position(self): return self.client.send_msg("p")
    def send_request_scoreboard(self): return self.client.send_msg("u")
    def send_goodbye(self): return self.client.send_msg("quit")
    def send_name(self, name): return self.client.send_msg(f"name;{name}")
    def send_say(self, msg): return self.client.send_msg(f"say;{msg}")
    def send_color(self, r, g, b): return self.client.send_msg(f"color;{r};{g};{b}")

    def _remember_async_observation(self, obs_name):
        obs_name = normalize_token(obs_name)
        if obs_name not in ("hit", "damage"):
            return
        with self.lock:
            if obs_name not in self.pending_observations:
                self.pending_observations.append(obs_name)
            if obs_name not in self.observations:
                self.observations.append(obs_name)
                self.latest_observation = Observation.from_tokens(self.observations)

    def _handle_message(self, cmd):
        if not cmd:
            return
        head = cmd[0].strip().lower()
        if head == "o":
            obs = parse_observation_payload(cmd[1] if len(cmd) > 1 else "")
            tokens = obs.as_list
            with self.lock:
                seen = set(tokens)
                for pending in self.pending_observations:
                    if pending not in seen:
                        tokens.append(pending)
                self.pending_observations.clear()
                self.latest_observation = Observation.from_tokens(tokens)
                self.observations = self.latest_observation.as_list
            self.obs_event.set()
            return
        if head == "s":
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
                self.log(f"[REDE] Status invalido: {cmd}")
            return
        if head == "p":
            try:
                with self.lock:
                    self.player_x = int(cmd[1])
                    self.player_y = int(cmd[2])
                    if len(cmd) > 3:
                        self.player_dir = cmd[3].lower()
            except (IndexError, ValueError):
                self.log(f"[REDE] Posicao invalida: {cmd}")
            return
        if head == "g":
            with self.lock:
                self.game_status = cmd[1] if len(cmd) > 1 else self.game_status
                if len(cmd) > 2:
                    try:
                        self.game_time = int(cmd[2])
                    except ValueError:
                        pass
            self.game_event.set()
            return
        if head == "u":
            with self.lock:
                self.scoreboard = cmd[1:]
            self.scoreboard_event.set()
            return
        if head == "notification":
            for token in normalize_tokens(cmd[1:]):
                if token in ("hit", "damage"):
                    self._remember_async_observation(token)
            self.log(f"[SERVIDOR] {';'.join(cmd[1:])}")
            return
        if head in ("h", "d"):
            self._remember_async_observation("hit" if head == "h" else "damage")
            return
        if head in ("hello", "goodbye"):
            self.log(f"[SERVIDOR] {';'.join(cmd)}")

    def request_observation_sync(self, timeout=1.0):
        self.obs_event.clear()
        if not self.send_request_observation():
            return None
        if not self.obs_event.wait(timeout):
            return None
        with self.lock:
            return list(self.observations)

    def request_status_sync(self, timeout=1.0):
        self.status_event.clear()
        if not self.send_request_user_status():
            return None
        if not self.status_event.wait(timeout):
            return None
        with self.lock:
            return (self.player_x, self.player_y, self.player_dir,
                    self.player_state, self.score, self.energy)

    def request_game_status_sync(self, timeout=0.5):
        self.game_event.clear()
        if not self.send_request_game_status():
            return None
        if not self.game_event.wait(timeout):
            return None
        with self.lock:
            return self.game_status, self.game_time

    def request_scoreboard_sync(self, timeout=0.5):
        self.scoreboard_event.clear()
        if not self.send_request_scoreboard():
            return None
        if not self.scoreboard_event.wait(timeout):
            return None
        with self.lock:
            return list(self.scoreboard)

    def request_sync_pair(self, timeout=0.5):
        self.status_event.clear()
        self.obs_event.clear()
        status_sent = self.send_request_user_status()
        obs_sent = self.send_request_observation()
        if not (status_sent and obs_sent):
            return None
        ok = self.status_event.wait(timeout) and self.obs_event.wait(timeout)
        if not ok:
            return None
        with self.lock:
            return (self.player_x, self.player_y, self.player_dir,
                    self.player_state, self.score, self.energy,
                    list(self.observations))
