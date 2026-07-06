"""Tactical combat controller."""

import time

from fuzzy import combat_aggressiveness
from world_model import BLOCKED, DANGEROUS, DIR_VECTORS, VISITED, SAFE, in_bounds

LOW_ENERGY = 40
CRITICAL_ENERGY = 25


class CombatController:
    def __init__(self, world, risk, profile="score", log=print):
        self.world = world
        self.risk = risk
        self.profile = profile
        self.log = log
        self.shots_since_hit = 0
        self.shots_fired = 0
        self.shots_hit = 0
        self.awaiting_shot_result = False
        self.engage_pause_until = 0.0
        self.hunt_reason = None
        self.hunt_turns = 0
        self._init_profile(profile)

    def _init_profile(self, profile):
        if profile == "aggressive":
            self.max_attack_dist = 9
            self.flee_aggr_min = 0.15
            self.steps_flee_aggr_min = 0.12
            self.damage_hunt_aggr_min = 0.30
            self.disengage_pause = 3.0
            self.initial_miss_limit = 4
            self.min_hit_rate_cutoff = 0.08
            self.min_shots_hitrate_check = 12
            self.hunt_turns_steps = 4
            self.hunt_turns_damage = 6
        elif profile == "safe":
            self.max_attack_dist = 4
            self.flee_aggr_min = 0.40
            self.steps_flee_aggr_min = 0.35
            self.damage_hunt_aggr_min = 0.70
            self.disengage_pause = 10.0
            self.initial_miss_limit = 2
            self.min_hit_rate_cutoff = 0.30
            self.min_shots_hitrate_check = 4
            self.hunt_turns_steps = 1
            self.hunt_turns_damage = 3
        else:
            self.max_attack_dist = 6
            self.flee_aggr_min = 0.30
            self.steps_flee_aggr_min = 0.25
            self.damage_hunt_aggr_min = 0.50
            self.disengage_pause = 8.0
            self.initial_miss_limit = 3
            self.min_hit_rate_cutoff = 0.20
            self.min_shots_hitrate_check = 6
            self.hunt_turns_steps = 2
            self.hunt_turns_damage = 4

    def reset(self):
        profile = self.profile
        self.__init__(self.world, self.risk, profile, self.log)

    def update_after_observation(self, observation):
        if self.awaiting_shot_result:
            self.shots_fired += 1
            if observation.hit:
                self.shots_hit += 1
                self.shots_since_hit = 0
                self.log("[COMBATE] Hit confirmado")
            else:
                self.shots_since_hit += 1
            self.awaiting_shot_result = False
        elif observation.hit:
            self.shots_hit += 1
            self.shots_since_hit = 0

    def aggressiveness(self, energy, observation):
        dist = observation.enemy_distance
        if dist is None:
            dist = 2 if (observation.damage or observation.steps) else 10
        return combat_aggressiveness(energy, dist)

    def line_of_fire_clear(self, x, y, direction, enemy_dist):
        if enemy_dist is None:
            return False
        vx, vy = DIR_VECTORS.get(direction, (0, 0))
        for step in range(1, enemy_dist + 1):
            tx, ty = x + vx * step, y + vy * step
            if not in_bounds(tx, ty):
                return False
            if self.world.grid[tx][ty] == BLOCKED:
                return False
        return True

    def miss_limit(self, enemy_dist):
        if enemy_dist is None:
            return 2
        if self.profile == "aggressive":
            if enemy_dist <= 4:
                return 12
            if enemy_dist <= 8:
                return 8
            return 5
        if self.profile == "safe":
            if enemy_dist <= 3:
                return 4
            return 2
        if enemy_dist <= 3:
            return 6
        if enemy_dist <= 6:
            return 4
        return 2

    def should_attack(self, energy, observation):
        enemy_dist = observation.enemy_distance
        if enemy_dist is None:
            return False
        if time.time() < self.engage_pause_until:
            return False
        if not self.line_of_fire_clear(*(self.world.position or (0, 0)),
                                       self.world.orientation, enemy_dist):
            return False
        if self.shots_hit == 0 and self.shots_fired >= self.initial_miss_limit:
            self.engage_pause_until = time.time() + self.disengage_pause
            return False
        if self.shots_since_hit >= self.miss_limit(enemy_dist):
            return False
        if energy <= CRITICAL_ENERGY and enemy_dist > 2:
            return False
        if enemy_dist > self.max_attack_dist and energy < 85:
            return False
        hit_rate = self.shots_hit / self.shots_fired if self.shots_fired else 0.5
        if self.shots_fired >= self.min_shots_hitrate_check and hit_rate < self.min_hit_rate_cutoff:
            return enemy_dist <= 3
        aggr = self.aggressiveness(energy, observation)
        if observation.hit:
            return energy > CRITICAL_ENERGY and enemy_dist <= self.max_attack_dist
        return aggr >= 0.42 or (energy >= 80 and enemy_dist <= 7)

    def decide(self, energy, observation):
        enemy_dist = observation.enemy_distance
        aggr = self.aggressiveness(energy, observation)
        if observation.damage:
            if energy <= LOW_ENERGY or aggr < self.damage_hunt_aggr_min:
                return "EVADE"
            self.hunt_reason = "damage"
            return "HUNT"
        if enemy_dist is not None:
            if self.should_attack(energy, observation):
                return "ATTACK"
            if enemy_dist <= 3 or aggr < self.flee_aggr_min or energy <= LOW_ENERGY:
                return "EVADE"
            if enemy_dist <= self.max_attack_dist + 3 and energy > LOW_ENERGY:
                return "CHASE"
        if observation.steps and aggr < self.steps_flee_aggr_min:
            return "FLEE"
        return None

    def record_shot(self):
        self.awaiting_shot_result = True

    def after_miss_limit(self, enemy_dist):
        if self.shots_since_hit >= self.miss_limit(enemy_dist):
            self.engage_pause_until = time.time() + self.disengage_pause
            self.shots_since_hit = 0
            return True
        return False

    def can_chase_forward(self, x, y, direction):
        vx, vy = DIR_VECTORS.get(direction, (0, 0))
        nx, ny = x + vx, y + vy
        if not in_bounds(nx, ny):
            return False
        cell = self.world.grid[nx][ny]
        if cell == BLOCKED or cell in DANGEROUS:
            return False
        if self.world.pit_risk(nx, ny) > 1:
            return False
        return self.world.is_walkable(nx, ny, allow_unknown=True)

    def best_evade_cell(self, x, y, direction, observation):
        candidates = []
        for dir_name, (vx, vy) in DIR_VECTORS.items():
            nx, ny = x + vx, y + vy
            if not in_bounds(nx, ny):
                continue
            cell = self.world.grid[nx][ny]
            if cell == BLOCKED or cell in DANGEROUS:
                continue
            if cell not in (SAFE, VISITED):
                continue
            same_axis = (
                (direction in ("north", "south") and dir_name in ("north", "south")) or
                (direction in ("east", "west") and dir_name in ("east", "west"))
            )
            score = 0.0
            if observation.damage or observation.enemy:
                score += 10.0 if not same_axis else -2.0
            if cell == VISITED:
                score += 3.0
            score += self.world.safe_exit_count(nx, ny)
            score -= self.risk.cell_penalty((nx, ny)) * 0.08
            candidates.append((score, (nx, ny)))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def hunt_turn_budget(self):
        return self.hunt_turns_steps if self.hunt_reason == "steps" else self.hunt_turns_damage
