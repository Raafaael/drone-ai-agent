"""Logger com nível: filtra o que é exibido sem precisar espalhar `if`s de
verbosidade pelo resto do código.

- "quiet": nada.
- "normal" (padrão): só as categorias relevantes durante uma partida
  ([FSM], [FARM], [COMBATE], [MAPA], [PLANO]...).
- "debug": tudo, sem filtro.
"""

import time


LEVELS = {"quiet": 0, "normal": 1, "debug": 2}

VISIBLE_AT_NORMAL = (
    "[INIT]", "[JOGO]", "[FIM]", "[REDE]", "[FSM]", "[COLETA]",
    "[FARM]", "[COMBATE]", "[STATUS]", "[MAPA]", "[PLANO]",
)


class AgentLogger:
    def __init__(self, level="normal", sink=print):
        self.level = LEVELS.get(level, 1)
        self.sink = sink

    def normal(self, msg):
        self.sink(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def __call__(self, msg):
        """Uso normal: `log("[FSM] ...")` — filtra pelo nível configurado."""
        if self.level == 0:
            return
        if self.level == 1 and not msg.startswith(VISIBLE_AT_NORMAL):
            return
        self.normal(msg)
