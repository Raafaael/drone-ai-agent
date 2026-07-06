"""Memória de ameaça e pontuação de risco de célula.

Duas noções de "perigo" convivem aqui, de propósito:

- `threat_until` reage a qualquer sinal de inimigo (passos, dano, avistado) e
  só serve para saber SE vale a pena reagir agora — decai rápido.
- `danger_cells` só registra evidência forte (dano de verdade, ou inimigo a
  3 células ou menos) e é o que de fato penaliza uma célula na hora de
  escolher rota/alvo. Se "passos" sozinho contasse aqui, um ruído de fundo
  qualquer envenenaria justamente os pontos de farm onde o agente mais fica
  parado.
"""

import time


THREAT_DECAY = 4.0
DANGER_MEMORY_WINDOW = 25.0
DANGER_RADIUS = 4
DANGER_PENALTY = 70.0


class RiskModel:
    """Calcula o quão arriscada é uma célula e escolhe destinos de fuga."""

    def __init__(self, world):
        self.world = world

    def reset(self):
        """Apaga a memória de ameaça (nova partida, mapa novo)."""
        self.world.danger_cells.clear()
        self.world.last_danger_cell = None
        self.world.threat_until = 0.0

    def update(self, position, observation, now=None):
        """Processa a observação do tick atual, atualizando as duas
        memórias de ameaça (ver docstring do módulo)."""
        now = now or time.time()
        enemy_dist = observation.enemy_distance
        if observation.steps or observation.damage or observation.enemy:
            self.world.threat_until = now + THREAT_DECAY
        if observation.damage or (enemy_dist is not None and enemy_dist <= 3):
            self.world.danger_cells[position] = now
            self.world.last_danger_cell = position
            self.world.version += 1
        self.expire(now)

    def expire(self, now=None):
        """Remove da memória as células de perigo mais antigas que a janela."""
        now = now or time.time()
        before = len(self.world.danger_cells)
        self.world.danger_cells = {
            cell: ts for cell, ts in self.world.danger_cells.items()
            if now - ts <= DANGER_MEMORY_WINDOW
        }
        if len(self.world.danger_cells) != before:
            self.world.version += 1

    def danger_penalty(self, cell, now=None):
        """Quão perto (no tempo e no espaço) `cell` está da ameaça mais
        forte conhecida. Usa a pior ameaça (não a soma), para uma célula com
        várias detecções antigas empilhadas não ficar punida demais."""
        now = now or time.time()
        worst = 0.0
        for pos, ts in self.world.danger_cells.items():
            age = now - ts
            if age > DANGER_MEMORY_WINDOW:
                continue
            dist = abs(pos[0] - cell[0]) + abs(pos[1] - cell[1])
            if dist > DANGER_RADIUS:
                continue
            recency = 1.0 - age / DANGER_MEMORY_WINDOW
            proximity = 1.0 - dist / (DANGER_RADIUS + 1)
            worst = max(worst, DANGER_PENALTY * recency * proximity)
        return worst

    def revisit_penalty(self, cell):
        """Penaliza células já muito visitadas (evita ficar circulando)."""
        return min(self.world.visit_count.get(cell, 0), 8) * 4.0

    def teleport_penalty(self, cell):
        return self.world.teleport_risk(*cell) * 12.0

    def pit_penalty(self, cell):
        risk = self.world.pit_risk(*cell)
        return 9999.0 if risk >= 99 else risk * 80.0

    def exit_bonus(self, cell):
        """Quantas saídas seguras a célula tem — bom sinal para fuga (evita
        becos sem saída)."""
        return self.world.safe_exit_count(*cell) * 8.0

    def cell_penalty(self, cell):
        """Custo de risco total de uma célula (poço + teleporte + revisita
        + ameaça recente), usado para pontuar rotas e alvos."""
        return (
            self.pit_penalty(cell)
            + self.teleport_penalty(cell)
            + self.revisit_penalty(cell)
            + self.danger_penalty(cell)
        )

    def flee_score(self, cell, current, energy=100):
        """Pontua `cell` como destino de fuga: distância da posição atual,
        saídas seguras, e principalmente o quanto ela afasta da ÚLTIMA
        célula de ameaça conhecida (fugir "para longe de onde eu estava"
        pode, sem essa checagem, passar perto de onde o tiro veio)."""
        dist = abs(cell[0] - current[0]) + abs(cell[1] - current[1])
        score = dist * 2.0 + self.exit_bonus(cell)
        if self.world.last_danger_cell is not None:
            dx, dy = self.world.last_danger_cell
            now_dist = abs(current[0] - dx) + abs(current[1] - dy)
            target_dist = abs(cell[0] - dx) + abs(cell[1] - dy)
            score += (target_dist - now_dist) * 6.0
        for pos, kind in self.world.item_spots.items():
            if kind == "powerup":
                score += max(0, 12 - abs(pos[0] - cell[0]) - abs(pos[1] - cell[1]))
        if energy < 35:
            score += self.exit_bonus(cell)
        return score - self.cell_penalty(cell)

    def allow_teleport_fallback(self, safe_plan_exists, stuck_or_looping):
        """Teleporte suspeito só é atravessado como último recurso: sem
        plano seguro, ou preso/repetindo o mesmo trecho do mapa."""
        return (not safe_plan_exists) or stuck_or_looping
