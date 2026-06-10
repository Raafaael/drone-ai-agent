"""
INF1771 - Trabalho Final: Desafio dos Drones
Ponto de entrada do agente.

Uso:
    python main.py [host] [nome]

Padrao: host = atari.icad.puc-rio.br (servidor de treino), nome = DroneIA.
"""

import sys
import time
import random

from devkit import GameAI
from ai_agent import DroneAgent

TICK = 0.07  # intervalo entre acoes (segundos)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "atari.icad.puc-rio.br"
    name = sys.argv[2] if len(sys.argv) > 2 else f"DroneIA_{random.randint(100, 999)}"

    ai = GameAI()
    log(f"[INIT] Conectando em {host}:8888 como '{name}'...")
    if not ai.connect(host, name):
        log("[INIT] Nao foi possivel conectar. Verifique o servidor.")
        sys.exit(1)

    ai.send_color(0, 200, 255)
    agent = DroneAgent(ai, log=log)
    log("[INIT] Conectado. Aguardando inicio da partida...")

    last_game_check = 0.0
    was_in_game = False
    try:
        while ai.connected:
            now = time.time()
            # verifica estado do jogo periodicamente
            if now - last_game_check > 3.0:
                ai.send_request_game_status()
                last_game_check = now

            status = ai.game_status.lower()
            in_game = "game" in status and "over" not in status

            # nova partida: zera o modelo de mundo (mapa/itens podem mudar)
            if in_game and not was_in_game:
                agent = DroneAgent(ai, log=log)
                log("[JOGO] Partida iniciada: novo modelo de mundo")
            was_in_game = in_game

            if in_game:
                if ai.player_state == "dead":
                    log("[JOGO] Drone morto. Aguardando proxima rodada...")
                    time.sleep(2)
                else:
                    agent.act()
            else:
                # Ready/Gameover: apenas espera (comandos de controle desabilitados)
                if "over" in status:
                    ai.send_request_scoreboard()
                    log(f"[JOGO] Estado: {ai.game_status} | pontos: {ai.score} | "
                        f"placar: {ai.scoreboard}")
                time.sleep(1)
                continue

            time.sleep(TICK)
    except KeyboardInterrupt:
        log("[FIM] Interrompido pelo usuario.")
    finally:
        log(f"[FIM] Pontuacao final: {ai.score}")
        ai.disconnect()


if __name__ == "__main__":
    main()
