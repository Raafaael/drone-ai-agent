"""Small configurable logger for the drone agent."""

import time


LEVELS = {"quiet": 0, "normal": 1, "debug": 2}


class AgentLogger:
    def __init__(self, level="normal", sink=print):
        self.level = LEVELS.get(level, 1)
        self.sink = sink

    def _emit(self, level, msg):
        if self.level >= level:
            self.sink(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def normal(self, msg):
        self._emit(1, msg)

    def debug(self, msg):
        self._emit(2, msg)

    def __call__(self, msg):
        if self.level == 0:
            return
        if self.level == 1 and not msg.startswith((
            "[INIT]", "[JOGO]", "[FIM]", "[REDE]", "[FSM]", "[COLETA]",
            "[FARM]", "[COMBATE]", "[STATUS]", "[MAPA]", "[PLANO]",
        )):
            return
        self.normal(msg)
