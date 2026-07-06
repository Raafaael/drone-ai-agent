"""Ponto de entrada: conecta no servidor e roda o loop da partida.

    python main.py [host] [nome]
"""

import argparse
import random
import sys
import time

from communication import GameAI
from logger import AgentLogger
from strategy import DroneAgent

DEFAULT_HOST = "atari.icad.puc-rio.br"
DEFAULT_COLOR = (255, 220, 0)
GAME_STATUS_POLL_INTERVAL = 3.0
SUMMARY_INTERVAL = 30.0


def parse_args(argv):
    parser = argparse.ArgumentParser(description="INF1771 drone AI agent")
    parser.add_argument("host", nargs="?", default=DEFAULT_HOST)
    parser.add_argument("name", nargs="?", default=None)
    parser.add_argument("--log-level", choices=("quiet", "normal", "debug"),
                        default="normal")
    parser.add_argument("--color", nargs=3, type=int, metavar=("R", "G", "B"),
                        default=DEFAULT_COLOR)
    return parser.parse_args(argv)


def log_summary(log, ai, agent):
    kinds = ", ".join(
        f"{k}={v}" for k, v in sorted(agent.farm.collect_by_kind.items())
    ) or "nenhuma"
    log(f"[STATUS] pontos={ai.score} energia={ai.energy} "
        f"pos=({ai.player_x},{ai.player_y}) dir={ai.player_dir} "
        f"estado={agent.state} coletas={agent.farm.collect_count} ({kinds}) "
        f"pontos_item={len(agent.world.item_spots)} metricas={agent.metrics_snapshot()}")


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    name = args.name or f"DroneIA_{random.randint(100, 999)}"
    log = AgentLogger(args.log_level)

    ai = GameAI(log=log)
    log(f"[INIT] Conectando em {args.host}:8888 como '{name}'")
    if not ai.connect(args.host, name):
        log("[INIT] Nao foi possivel conectar. Verifique o servidor.")
        return 1
    color = tuple(max(0, min(255, c)) for c in args.color)
    ai.send_color(*color)
    agent = DroneAgent(ai, log=log)
    log("[INIT] Conectado. Aguardando inicio da partida.")

    last_game_check = 0.0
    last_summary = 0.0
    was_in_game = False
    try:
        while ai.connected:
            now = time.time()
            if now - last_game_check > GAME_STATUS_POLL_INTERVAL:
                ai.send_request_game_status()
                last_game_check = now

            status = ai.game_status.lower()
            in_game = "game" in status and "over" not in status
            if in_game and not was_in_game:
                agent = DroneAgent(ai, log=log)
                # player_state so atualiza quando chega uma mensagem 's' do
                # servidor (dentro de agent.act()). Se o drone morreu na
                # partida anterior e nunca chamassemos act() de novo, ele
                # ficaria travado achando que ainda esta morto.
                ai.player_state = "game"
                last_summary = 0.0
                log("[JOGO] Partida iniciada: modelo reiniciado")
            was_in_game = in_game

            if not in_game:
                if "over" in status:
                    scoreboard = ai.request_scoreboard_sync(timeout=0.4)
                    log(f"[JOGO] Estado={ai.game_status} pontos={ai.score} placar={scoreboard}")
                time.sleep(1)
                continue

            if ai.player_state == "dead":
                log("[JOGO] Drone morto. Aguardando proxima rodada.")
                time.sleep(2)
                continue

            agent.act()
            if now - last_summary >= SUMMARY_INTERVAL:
                log_summary(log, ai, agent)
                last_summary = now
    except KeyboardInterrupt:
        log("[FIM] Interrompido pelo usuario.")
    finally:
        log(f"[FIM] Metricas finais: {agent.metrics_snapshot()}")
        log(f"[FIM] Pontuacao final: {ai.score}")
        ai.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
