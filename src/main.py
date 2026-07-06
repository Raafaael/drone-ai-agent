"""Executable entry point for the consolidated drone agent."""

import argparse
import random
import sys
import time

from communication import GameAI
from logger import AgentLogger
from strategy import DroneAgent

DEFAULT_HOST = "atari.icad.puc-rio.br"
# strategy.sync() ja bloqueia pelo tempo real de rede (Event.wait) sempre que
# uma acao (forward/turn/shoot/grab) marca a proxima observacao/status como
# necessaria -- ou seja, na quase totalidade dos ticks ativos. Somar uma
# espera fixa aqui em cima disso e puro atraso duplicado por acao (mesma
# classe de bug ja corrigida uma vez no TICK antigo). Os valores abaixo sao
# so uma rede de seguranca contra spin de CPU nos ticks em que o cache ainda
# e valido (ex.: CAMP esperando respawn sem rede nenhuma), nao um limitador
# de velocidade de acao.
NORMAL_TICK = 0.0
FAST_TICK = 0.0
OPENING_TICK = 0.0
OPENING_SECONDS = 25.0
SUMMARY_INTERVAL = 30.0
ACTIVE_STATES = {"ATTACK", "CHASE", "EVADE", "HUNT", "FLEE"}


def parse_args(argv):
    parser = argparse.ArgumentParser(description="INF1771 drone AI agent")
    parser.add_argument("host", nargs="?", default=DEFAULT_HOST)
    parser.add_argument("name", nargs="?", default=None)
    parser.add_argument("--profile", choices=("score", "aggressive", "safe"),
                        default="score")
    parser.add_argument("--aggressive", "--aggresive", action="store_true",
                        help="alias for --profile aggressive")
    parser.add_argument("--log-level", choices=("quiet", "normal", "debug"),
                        default="normal")
    parser.add_argument("--color", nargs=3, type=int, metavar=("R", "G", "B"),
                        default=(255, 220, 0))
    return parser.parse_args(argv)


def action_delay(agent, game_started_at):
    if game_started_at is None:
        return NORMAL_TICK
    if agent.state in ACTIVE_STATES:
        return FAST_TICK
    if time.time() - game_started_at <= OPENING_SECONDS:
        return agent.config.get("early_tick", OPENING_TICK)
    return NORMAL_TICK


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
    profile = "aggressive" if args.aggressive else args.profile
    name = args.name or f"DroneIA_{random.randint(100, 999)}"
    logger = AgentLogger(args.log_level)
    log = logger

    ai = GameAI(log=log)
    log(f"[INIT] Conectando em {args.host}:8888 como '{name}' perfil={profile}")
    if not ai.connect(args.host, name):
        log("[INIT] Nao foi possivel conectar. Verifique o servidor.")
        return 1
    color = tuple(max(0, min(255, c)) for c in args.color)
    ai.send_color(*color)
    agent = DroneAgent(ai, log=log, profile=profile)
    log("[INIT] Conectado. Aguardando inicio da partida.")

    last_game_check = 0.0
    last_summary = 0.0
    was_in_game = False
    game_started_at = None
    try:
        while ai.connected:
            now = time.time()
            if now - last_game_check > 3.0:
                ai.send_request_game_status()
                last_game_check = now

            status = ai.game_status.lower()
            in_game = "game" in status and "over" not in status
            if in_game and not was_in_game:
                agent = DroneAgent(ai, log=log, profile=profile)
                # player_state so e atualizado quando chega uma mensagem 's'
                # do servidor, e isso so acontece dentro de agent.act(). Se o
                # drone morreu na partida anterior, o loop nunca mais chama
                # agent.act() enquanto achar que esta 'dead' -- travando para
                # sempre, mesmo com a partida nova ja em andamento e o drone
                # vivo de novo. Reseta aqui porque o estado antigo nao vale
                # mais para a partida que acabou de comecar.
                ai.player_state = "game"
                game_started_at = now
                last_summary = 0.0
                log("[JOGO] Partida iniciada: modelo reiniciado")
            was_in_game = in_game

            if in_game:
                if ai.player_state == "dead":
                    log("[JOGO] Drone morto. Aguardando proxima rodada.")
                    time.sleep(2)
                    continue
                agent.act()
                if now - last_summary >= SUMMARY_INTERVAL:
                    log_summary(log, ai, agent)
                    last_summary = now
                time.sleep(action_delay(agent, game_started_at))
            else:
                game_started_at = None
                if "over" in status:
                    scoreboard = ai.request_scoreboard_sync(timeout=0.4)
                    log(f"[JOGO] Estado={ai.game_status} pontos={ai.score} placar={scoreboard}")
                time.sleep(1)
    except KeyboardInterrupt:
        log("[FIM] Interrompido pelo usuario.")
    finally:
        if "agent" in locals():
            log(f"[FIM] Metricas finais: {agent.metrics_snapshot()}")
        log(f"[FIM] Pontuacao final: {ai.score}")
        ai.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
