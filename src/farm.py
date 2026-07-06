"""Economic item collection and respawn farming strategy."""

import time

from risk import DANGER_PENALTY
from world_model import SAFE, UNKNOWN, neighbors

SEC_PER_ACTION = 0.12
RESPAWN_DEFAULT = 12.0
ITEM_VALUE = {"treasure": 1000.0, "unknown": 700.0, "powerup": 300.0}
EXPLORE_BASE = 900.0
EARLY_EXPLORE_SECONDS = 60.0
EARLY_EXPLORE_MIN_SPOTS = 3
EARLY_EXPLORE_BOOST = 0.60
EARLY_FARM_DISCOUNT = 0.25
FRONTIER_INFO_WEIGHT = 0.15
PENDING_GRAB_TIMEOUT = 1.2
POWERUP_PICKUP_ENERGY = 70
# A escolha de alvo penalizava risco com um desconto FIXO (cell_penalty*0.05,
# no maximo ~3.5 de uma ameaca recente de ate 70): irrelevante frente a
# utilidades na casa das centenas/milhares, e mesmo corrigido para um peso
# maior continuaria irrelevante, porque farm_utility() (valor/segundo) cresce
# muito rapido para alvos proximos -- um alvo 1 passo mais perto ja vale
# centenas de pontos a mais, o que nenhum desconto fixo de ate 70 supera.
# Por isso o desconto por ameaca recente agora e MULTIPLICATIVO (corta uma
# fracao do valor do proprio alvo, nao um numero fixo), e so risco de
# poco/teleporte/revisita continua como desconto fixo (esses nao dependem de
# comparar com um alvo mais proximo, so afinam a rota).
# Sobrevivencia importa em toda partida (o enunciado encerra a partida do
# agente ao morrer, nao so desconta -10), entao ameaca recente precisa pesar
# de verdade na escolha do alvo, nao so na fuga reativa.
DANGER_AVOIDANCE_FRACTION = 0.85
ROUTE_PENALTY_WEIGHT = 0.3


class FarmManager:
    def __init__(self, world, planner, risk, log=print):
        self.world = world
        self.planner = planner
        self.risk = risk
        self.log = log
        self.start_time = time.time()
        self.last_taken = {}
        self.last_grab = {}
        self.spot_respawn = {}
        self.spot_next_check = {}
        self.spot_misses = {}
        self.item_values = {}
        self.pending_grabs = []
        self.collect_count = 0
        self.collect_by_kind = {}

    def reset(self):
        self.__init__(self.world, self.planner, self.risk, self.log)

    def known_treasure_spots(self):
        return sum(1 for k in self.world.item_spots.values()
                   if k in ("treasure", "unknown"))

    def early_explore_pressure(self, now=None):
        now = now or time.time()
        known = self.known_treasure_spots()
        time_left = max(0.0, 1.0 - (now - self.start_time) / EARLY_EXPLORE_SECONDS)
        item_gap = max(0.0, (EARLY_EXPLORE_MIN_SPOTS - known) / EARLY_EXPLORE_MIN_SPOTS)
        return max(time_left, 0.8 * item_gap)

    def frontier_info_gain(self, pos):
        unknown_adj = sum(1 for n in neighbors(*pos) if self.world.grid[n[0]][n[1]] == UNKNOWN)
        safe_adj = sum(1 for n in neighbors(*pos) if self.world.grid[n[0]][n[1]] == SAFE)
        return 1.0 + unknown_adj + 0.5 * safe_adj

    def due_spots(self, kinds, energy, now=None):
        now = now or time.time()
        out = []
        for pos, kind in self.world.item_spots.items():
            if kind not in kinds:
                continue
            if kind == "powerup" and energy > POWERUP_PICKUP_ENERGY:
                continue
            if now < self.spot_next_check.get(pos, 0):
                continue
            estimate = self.spot_respawn.get(pos, RESPAWN_DEFAULT)
            if pos not in self.last_taken or now - self.last_taken[pos] >= estimate:
                out.append(pos)
        return out

    def farm_utility(self, pos, kind, dist_steps, now=None):
        now = now or time.time()
        respawn = self.spot_respawn.get(pos, RESPAWN_DEFAULT)
        last = self.last_taken.get(pos)
        ripe_in = 0.0 if last is None else max(0.0, respawn - (now - last))
        arrive = dist_steps * SEC_PER_ACTION
        wait = max(0.0, ripe_in - arrive)
        total = arrive + wait + SEC_PER_ACTION
        learned = self.item_values.get(pos)
        value = learned or ITEM_VALUE.get(kind, 500.0)
        return value / max(total, SEC_PER_ACTION)

    def has_collectable_item(self, pos, observation, energy, now=None):
        now = now or time.time()
        if not observation.has_item:
            return False
        if observation.light == "powerup" and energy > POWERUP_PICKUP_ENERGY:
            return False
        return now - self.last_grab.get(pos, 0) > 0.6

    def record_grab(self, pos, score, now=None):
        now = now or time.time()
        kind = self.world.item_spots.get(pos, self.world.items.get(pos, "item"))
        self.world.consume_item(*pos)
        self.last_taken[pos] = now
        self.last_grab[pos] = now
        self.pending_grabs.append((pos, score, now))
        self.collect_count += 1
        self.collect_by_kind[kind] = self.collect_by_kind.get(kind, 0) + 1
        self.spot_next_check[pos] = now + self.spot_respawn.get(pos, RESPAWN_DEFAULT)
        self.spot_misses[pos] = 0
        self.log(f"[COLETA] {kind} em {pos} | total={self.collect_count}")

    def resolve_pending_grabs(self, score, now=None):
        now = now or time.time()
        ordered = sorted(self.pending_grabs, key=lambda item: item[2])
        still = []
        for idx, (pos, prev_score, grabbed_at) in enumerate(ordered):
            reference = ordered[idx + 1][1] if idx + 1 < len(ordered) else score
            if reference != prev_score or now - grabbed_at > PENDING_GRAB_TIMEOUT:
                delta = reference - prev_score
                if delta > 0:
                    old = self.item_values.get(pos)
                    self.item_values[pos] = delta if old is None else int(old * 0.7 + delta * 0.3)
                    self.log(f"[FARM] Valor aprendido em {pos}: +{self.item_values[pos]}")
            else:
                still.append((pos, prev_score, grabbed_at))
        self.pending_grabs = still

    def observe_respawn(self, pos, observation, now=None):
        now = now or time.time()
        if pos not in self.world.item_spots:
            return
        if pos not in self.last_taken:
            if not observation.has_item:
                misses = self.spot_misses.get(pos, 0) + 1
                self.spot_misses[pos] = misses
                self.spot_next_check[pos] = now + min(8.0, 1.5 * misses)
            else:
                self.spot_misses[pos] = 0
                self.spot_next_check[pos] = now
            return
        age = now - self.last_taken[pos]
        cur = self.spot_respawn.get(pos, RESPAWN_DEFAULT)
        if observation.has_item and age < cur:
            self.spot_respawn[pos] = max(2.0, age)
            self.spot_next_check[pos] = now
            self.spot_misses[pos] = 0
            self.log(f"[FARM] {pos} respawn mais rapido: {self.spot_respawn[pos]:.0f}s")
        elif not observation.has_item and cur < age < 120:
            misses = self.spot_misses.get(pos, 0) + 1
            self.spot_misses[pos] = misses
            self.spot_respawn[pos] = min(60.0, age + 1.5)
            self.spot_next_check[pos] = now + min(8.0, 1.5 * misses)

    def _risk_adjusted_value(self, value, pos):
        """Aplica o desconto de risco a um valor de alvo ja calculado:
        ameaca recente corta uma FRACAO do proprio valor (proporcional,
        nao um numero fixo -- ver DANGER_AVOIDANCE_FRACTION); poco/teleporte/
        revisita continuam como desconto fixo, so para afinar a rota."""
        danger = self.risk.danger_penalty(pos)
        if danger:
            value *= max(0.0, 1.0 - (danger / DANGER_PENALTY) * DANGER_AVOIDANCE_FRACTION)
        value -= (self.risk.teleport_penalty(pos) + self.risk.revisit_penalty(pos)) * ROUTE_PENALTY_WEIGHT
        return value

    def next_check_at(self, pos, now=None):
        now = now or time.time()
        if pos not in self.world.item_spots:
            return None
        if pos in self.spot_next_check:
            return self.spot_next_check[pos]
        if pos not in self.last_taken:
            return now
        return self.last_taken[pos] + self.spot_respawn.get(pos, RESPAWN_DEFAULT)

    def choose_plan(self, start, direction, energy, allow_flash=False):
        now = time.time()
        dist, parent = self.planner.reachable_map(start, allow_flash=allow_flash)
        frontier = self.world.frontier_cells()
        early = self.early_explore_pressure(now)
        best = None

        for pos, kind in self.world.item_spots.items():
            if kind == "powerup" and energy > POWERUP_PICKUP_ENERGY:
                continue
            if now < self.spot_next_check.get(pos, 0):
                continue
            dd = 0 if pos == start else dist.get(pos)
            if dd is None:
                continue
            value = self.farm_utility(pos, kind, dd, now)
            if pos in self.last_taken:
                value *= 1.0 - EARLY_FARM_DISCOUNT * early
            value -= self.spot_misses.get(pos, 0) * 35
            value = self._risk_adjusted_value(value, pos)
            if best is None or value > best[0]:
                best = (value, pos, "FARM")

        known = self.known_treasure_spots()
        explore_value = EXPLORE_BASE * (1.0 + EARLY_EXPLORE_BOOST * early) / (1.0 + known)
        for pos in frontier:
            dd = dist.get(pos)
            if dd is None:
                continue
            info = self.frontier_info_gain(pos)
            value = explore_value * (1.0 + FRONTIER_INFO_WEIGHT * info) / (dd * SEC_PER_ACTION + 1.0)
            value = self._risk_adjusted_value(value, pos)
            if best is None or value > best[0]:
                best = (value, pos, "PLANO")

        if best is None:
            return None
        utility, goal, label = best
        if goal == start:
            return {"label": "CAMP", "goal": goal, "path": [], "utility": utility}
        path = self.planner.a_star(start, goal, start_dir=direction,
                                   allow_unknown=True, allow_flash=allow_flash)
        if path is None:
            path = self.planner.path_from_parent(parent, goal)
        if path is None:
            return None
        return {"label": label, "goal": goal, "path": path, "utility": utility}

    def has_better_reachable_plan(self, start, direction, energy, margin=1.20):
        current_kind = self.world.item_spots.get(start)
        if current_kind not in ("treasure", "unknown"):
            return False
        now = time.time()
        current_value = self.farm_utility(start, current_kind, 0, now)
        plan = self.choose_plan(start, direction, energy, allow_flash=False)
        if plan is None or plan["label"] == "CAMP" or plan["goal"] == start:
            return False
        return plan["utility"] > current_value * margin
