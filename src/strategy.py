"""Behavior-first finite-state controller for the drone agent."""

from collections import Counter
from dataclasses import dataclass, field
import time

from combat import CombatController, LOW_ENERGY
from farm import FarmManager
from observations import Observation
from planner import Planner
from risk import RiskModel
from world_model import (
    BLOCKED,
    DANGER_FLASH,
    DANGEROUS,
    DIR_VECTORS,
    UNKNOWN,
    VISITED,
    WorldModel,
    in_bounds,
    neighbors,
)

TURN_LEFT_OF = {"north": "west", "west": "south", "south": "east", "east": "north"}
TURN_RIGHT_OF = {"north": "east", "east": "south", "south": "west", "west": "north"}

PROFILE_CONFIG = {
    "score": {
        "status_interval": 1.2,
        "observation_min_interval": 0.45,
        "observation_ttl": 3.0,
        "enemy_ttl": 1.2,
        "max_chase_ticks": 3,
        "blocked_chase_cooldown": 6.0,
        "max_blocked_chase_failures": 1,
        "max_reposition_steps": 2,
        "respawn_check_window": 0.8,
    },
    "aggressive": {
        "status_interval": 0.8,
        "observation_min_interval": 0.28,
        "observation_ttl": 2.0,
        "enemy_ttl": 1.5,
        "max_chase_ticks": 6,
        "blocked_chase_cooldown": 4.0,
        "max_blocked_chase_failures": 2,
        "max_reposition_steps": 3,
        "respawn_check_window": 0.6,
    },
    "safe": {
        "status_interval": 1.5,
        "observation_min_interval": 0.6,
        "observation_ttl": 2.5,
        "enemy_ttl": 0.9,
        "max_chase_ticks": 2,
        "blocked_chase_cooldown": 8.0,
        "max_blocked_chase_failures": 1,
        "max_reposition_steps": 1,
        "respawn_check_window": 1.0,
    },
}


@dataclass
class AgentMetrics:
    decisions: int = 0
    commands: Counter = field(default_factory=Counter)
    sync_pairs: int = 0
    status_requests: int = 0
    observation_requests: int = 0
    moves: int = 0
    rotations: int = 0
    grabs: int = 0
    shots: int = 0
    hits: int = 0
    damage: int = 0
    idle_without_command: int = 0
    routes_cancelled: int = 0
    loops_detected: int = 0
    plan_reused: int = 0
    blocked_chase_failures: int = 0
    repeated_blocks: int = 0
    repositions: int = 0
    chases_abandoned: int = 0
    previous_plan_restored: int = 0
    fsm_oscillations: int = 0
    transitions: Counter = field(default_factory=Counter)
    started_at: float = field(default_factory=time.time)

    def record_command(self, command):
        self.commands[command] += 1
        if command in ("w", "s"):
            self.moves += 1
        elif command in ("a", "d"):
            self.rotations += 1
        elif command == "t":
            self.grabs += 1
        elif command == "e":
            self.shots += 1

    def snapshot(self, score=None):
        elapsed = max(0.001, time.time() - self.started_at)
        total = sum(self.commands.values())
        return {
            "total_commands": total,
            "commands": dict(self.commands),
            "observations_o": self.commands.get("o", 0),
            "status_q": self.commands.get("q", 0),
            "moves": self.moves,
            "rotations": self.rotations,
            "grabs": self.grabs,
            "shots": self.shots,
            "hits": self.hits,
            "damage": self.damage,
            "shots_per_hit": None if self.hits == 0 else round(self.shots / self.hits, 2),
            "observations_per_min": round(self.commands.get("o", 0) * 60 / elapsed, 2),
            "actions_per_min": round(total * 60 / elapsed, 2),
            "idle_without_command": self.idle_without_command,
            "routes_cancelled": self.routes_cancelled,
            "loops_detected": self.loops_detected,
            "plan_reused": self.plan_reused,
            "blocked_chase_failures": self.blocked_chase_failures,
            "repeated_blocks": self.repeated_blocks,
            "repositions": self.repositions,
            "chases_abandoned": self.chases_abandoned,
            "previous_plan_restored": self.previous_plan_restored,
            "fsm_oscillations": self.fsm_oscillations,
            "transitions": dict(self.transitions),
            "score": score,
        }


class DroneAgent:
    """Executable AI agent with economical observation policy."""

    STATES = {
        "EXPLORE", "GRAB", "RECHARGE", "ATTACK", "CHASE", "EVADE",
        "HUNT", "FLEE", "CAMP", "REPOSITION",
    }

    def __init__(self, game_ai, log=print, profile="score", aggressive=False):
        if aggressive:
            profile = "aggressive"
        self.ai = game_ai
        self.log = log
        self.profile = profile
        self.config = PROFILE_CONFIG.get(profile, PROFILE_CONFIG["score"])
        self.world = WorldModel()
        self.risk = RiskModel(self.world)
        self.planner = Planner(self.world, self.risk)
        self.farm = FarmManager(self.world, self.planner, self.risk, log=log)
        self.combat = CombatController(self.world, self.risk, profile=profile, log=log)

        self.state = "EXPLORE"
        self.path = []
        self.path_allows_flash = False
        self.goal = None
        self.last_action = "wait"
        self.last_target = None
        self.last_pos = None
        self.forward_fails = 0
        self.sync_fails = 0
        self.tick_count = 0
        self.idle_since = None
        self.state_entered_at = time.time()
        self.metrics = AgentMetrics()

        self._cached_view = None
        self._last_status_at = 0.0
        self._last_observation_at = 0.0
        self._need_status = True
        self._need_observation = True
        self._chase_ticks = 0
        self._last_camp_log_at = 0.0
        self._enemy_seen_at = 0.0
        self._enemy_seen_pose = None
        self._enemy_seen_distance = None
        self._last_enemy_observation_timestamp = 0.0
        self._blocked_chase_failures = {}
        self._blocked_chase_cooldown = {}
        self._failed_reposition_cells = set()
        self._fsm_history = []
        self._previous_plan = None
        self._reposition_steps_left = 0

    def reset_for_new_game(self):
        self.__init__(self.ai, log=self.log, profile=self.profile)

    def metrics_snapshot(self):
        score = getattr(self.ai, "score", None)
        return self.metrics.snapshot(score=score)

    # ---------------- economical communication ----------------

    def _record_sent(self, command):
        self.metrics.record_command(command)

    def _request_status_next(self):
        self._need_status = True

    def _request_observation_next(self):
        self._need_observation = True

    def _merge_events(self, events):
        if not events or self._cached_view is None:
            return
        x, y, direction, pstate, score, energy, obs = self._cached_view
        tokens = list(obs)
        for event in events:
            if event not in tokens:
                tokens.append(event)
        self._cached_view = (x, y, direction, pstate, score, energy, tokens)

    def sync(self):
        now = time.time()
        timeout = 0.45 + min(self.sync_fails * 0.18, 1.2)

        if hasattr(self.ai, "consume_pending_events"):
            self._merge_events(self.ai.consume_pending_events())

        if self._cached_view is None:
            view = self.ai.request_sync_pair(timeout=timeout)
            self.metrics.sync_pairs += 1
            self._record_sent("q")
            self._record_sent("o")
            if view is not None:
                self._cached_view = view
                self._last_status_at = now
                self._last_observation_at = now
                self._need_status = False
                self._need_observation = False
            return view

        obs_age = now - self._last_observation_at
        status_age = now - self._last_status_at
        urgent_obs = self._need_observation and self.last_action in {
            "forward", "turn", "shoot", "grab",
        }
        obs_due = (
            self._need_observation
            and (urgent_obs or obs_age >= self.config["observation_min_interval"])
        ) or obs_age >= self.config["observation_ttl"]
        status_due = self._need_status or status_age >= self.config["status_interval"]

        if status_due and obs_due:
            view = self.ai.request_sync_pair(timeout=timeout)
            self.metrics.sync_pairs += 1
            self._record_sent("q")
            self._record_sent("o")
            if view is None:
                return None
            self._cached_view = view
            self._last_status_at = now
            self._last_observation_at = now
            self._need_status = False
            self._need_observation = False
            return view

        if status_due and hasattr(self.ai, "request_status_sync"):
            status = self.ai.request_status_sync(timeout=timeout)
            self.metrics.status_requests += 1
            self._record_sent("q")
            if status is None:
                return None
            x, y, direction, pstate, score, energy = status
            _, _, _, _, _, _, obs = self._cached_view
            self._cached_view = (x, y, direction, pstate, score, energy, obs)
            self._last_status_at = now
            self._need_status = False

        if obs_due and hasattr(self.ai, "request_observation_sync"):
            obs = self.ai.request_observation_sync(timeout=timeout)
            self.metrics.observation_requests += 1
            self._record_sent("o")
            if obs is None:
                return None
            x, y, direction, pstate, score, energy, _ = self._cached_view
            self._cached_view = (x, y, direction, pstate, score, energy, obs)
            self._last_observation_at = now
            self._need_observation = False

        return self._cached_view

    # ---------------- low-level actions ----------------

    def _set_state(self, new_state, reason=None):
        if new_state not in self.STATES:
            raise ValueError(f"estado desconhecido: {new_state}")
        if new_state != self.state:
            old_state = self.state
            self.log(f"[FSM] {self.state} -> {new_state}" + (f" | {reason}" if reason else ""))
            self.metrics.transitions[f"{old_state}->{new_state}"] += 1
            self._remember_state_transition(old_state, new_state, reason)
            if old_state not in {"ATTACK", "CHASE", "HUNT", "EVADE", "REPOSITION", "FLEE"} \
                    and new_state in {"ATTACK", "CHASE", "HUNT", "EVADE", "REPOSITION", "FLEE"}:
                self._save_previous_plan()
            if "HUNT" not in (self.state, new_state) and new_state != "REPOSITION":
                self.path = []
            if new_state != "CHASE":
                self._chase_ticks = 0
            self.state = new_state
            self.state_entered_at = time.time()

    def _save_previous_plan(self):
        if self._previous_plan is not None:
            return
        if self.path:
            self._previous_plan = {
                "goal": self.goal,
                "path": list(self.path),
                "allows_flash": self.path_allows_flash,
                "saved_at": time.time(),
            }

    def _restore_previous_plan(self):
        if not self._previous_plan:
            return False
        path = list(self._previous_plan["path"])
        if not path:
            return False
        for x, y in path:
            cell = self.world.grid[x][y]
            if cell == BLOCKED or cell in DANGEROUS:
                return False
        self.goal = self._previous_plan["goal"]
        self.path = path
        self.path_allows_flash = self._previous_plan["allows_flash"]
        self.metrics.previous_plan_restored += 1
        self.log(f"[PLANO] Retomando plano anterior para {self.goal}")
        return True

    def _remember_state_transition(self, old_state, new_state, reason):
        pos = getattr(self, "_current_pos", None)
        direction = getattr(self, "_current_direction", None)
        enemy_dist = self._enemy_seen_distance
        blocked = self._front_cell(*pos, direction) if pos and direction else None
        self._fsm_history.append({
            "old": old_state,
            "new": new_state,
            "pos": pos,
            "direction": direction,
            "enemy": enemy_dist,
            "blocked": blocked,
            "action": self.last_action,
            "reason": reason,
            "time": time.time(),
        })
        if len(self._fsm_history) > 12:
            self._fsm_history.pop(0)

    def _oscillating_chase_hunt(self):
        recent = self._fsm_history[-4:]
        if len(recent) < 4:
            return False
        pattern = [f"{r['old']}->{r['new']}" for r in recent]
        if pattern != ["CHASE->HUNT", "HUNT->CHASE", "CHASE->HUNT", "HUNT->CHASE"]:
            return False
        same_pos = len({r["pos"] for r in recent}) == 1
        same_block = len({r["blocked"] for r in recent}) == 1
        return same_pos and same_block

    def _front_cell(self, x, y, direction):
        dx, dy = DIR_VECTORS.get(direction, (0, 0))
        return x + dx, y + dy

    def _left_right_dirs(self, direction):
        return TURN_LEFT_OF[direction], TURN_RIGHT_OF[direction]

    def face(self, current_dir, target_dir):
        if current_dir == target_dir:
            return True
        if TURN_LEFT_OF[current_dir] == target_dir:
            self.ai.send_turn_left()
            self._record_sent("a")
            self.log(f"[ACAO] Virar a esquerda ({current_dir} -> {target_dir})")
        else:
            self.ai.send_turn_right()
            self._record_sent("d")
            self.log(f"[ACAO] Virar a direita ({current_dir} -> {target_dir})")
        self.last_action = "turn"
        self.last_target = None
        self._request_status_next()
        self._request_observation_next()
        return False

    def step_towards(self, x, y, direction, target):
        dx, dy = target[0] - x, target[1] - y
        want = None
        for dir_name, vec in DIR_VECTORS.items():
            if vec == (dx, dy):
                want = dir_name
                break
        if want is None:
            self.path = []
            self.last_action = "wait"
            self.last_target = None
            return
        if self.face(direction, want):
            self.ai.send_forward()
            self._record_sent("w")
            self.last_action = "forward"
            self.last_target = target
            self._request_status_next()
            self._request_observation_next()
            self.log(f"[ACAO] Andar para frente -> {target}")

    # ---------------- perception and decision ----------------

    def _handle_blockage_and_motion(self, x, y, direction, observation):
        if observation.blocked:
            bx, by = self._front_cell(x, y, direction)
            if in_bounds(bx, by):
                if self.world.grid[bx][by] == BLOCKED:
                    self.metrics.repeated_blocks += 1
                self.world.mark_blocked(bx, by)
                self.planner.invalidate()
                self.log(f"[MAPA] Posicao bloqueada em ({bx},{by})")
            self.metrics.routes_cancelled += 1
            self.path = []
            self._request_observation_next()

        if self.last_action == "forward" and self.last_pos == (x, y) and not observation.blocked:
            self.forward_fails += 1
            if self.forward_fails >= 3:
                bx, by = self._front_cell(x, y, direction)
                if in_bounds(bx, by):
                    if self.world.grid[bx][by] == BLOCKED:
                        self.metrics.repeated_blocks += 1
                    self.world.mark_blocked(bx, by)
                    self.planner.invalidate()
                    self.log(f"[MAPA] Travado: assumindo bloqueio em ({bx},{by})")
                self.metrics.routes_cancelled += 1
                self.path = []
                self.forward_fails = 0
        elif self.last_pos != (x, y):
            if self.last_action == "forward" and self.last_pos and self.last_target:
                jumped = abs(x - self.last_pos[0]) + abs(y - self.last_pos[1]) > 1
                if jumped:
                    self.log(f"[MAPA] Teleporte provavel em {self.last_target} -> ({x},{y})")
                    self.path = []
                    self.metrics.routes_cancelled += 1
            self.forward_fails = 0
        self.last_pos = (x, y)

    def _looping(self):
        recent = self.world.recent_positions
        if len(recent) < 24:
            return False
        xs = [p[0] for p in recent[-24:]]
        ys = [p[1] for p in recent[-24:]]
        area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        looping = area <= 24 and len(set(recent[-24:])) <= 12
        if looping:
            self.metrics.loops_detected += 1
        return looping

    def _update_enemy_memory(self, x, y, direction, observation):
        if observation.enemy:
            if self._last_observation_at != self._last_enemy_observation_timestamp:
                self._enemy_seen_at = self._last_observation_at or time.time()
                self._last_enemy_observation_timestamp = self._last_observation_at
            elif self._enemy_seen_at == 0.0:
                self._enemy_seen_at = time.time()
            self._enemy_seen_pose = (x, y, direction)
            self._enemy_seen_distance = observation.enemy_distance

    def _enemy_fresh(self, observation):
        if not observation.enemy:
            return False
        return time.time() - self._enemy_seen_at <= self.config["enemy_ttl"]

    def _chase_key(self, x, y, direction, enemy_dist):
        blocked = self._front_cell(x, y, direction)
        return ((x, y), direction, blocked, enemy_dist)

    def _chase_in_cooldown(self, x, y, direction, enemy_dist):
        key = self._chase_key(x, y, direction, enemy_dist)
        until = self._blocked_chase_cooldown.get(key, 0.0)
        if time.time() < until:
            return True
        self._blocked_chase_cooldown.pop(key, None)
        return False

    def _record_blocked_chase(self, x, y, direction, enemy_dist):
        blocked = self._front_cell(x, y, direction)
        if in_bounds(*blocked):
            if self.world.grid[blocked[0]][blocked[1]] == BLOCKED:
                self.metrics.repeated_blocks += 1
            self.world.mark_blocked(*blocked)
            self.planner.invalidate()
        key = self._chase_key(x, y, direction, enemy_dist)
        failures = self._blocked_chase_failures.get(key, 0) + 1
        self._blocked_chase_failures[key] = failures
        cooldown = self.config["blocked_chase_cooldown"] * failures
        self._blocked_chase_cooldown[key] = time.time() + cooldown
        self.metrics.blocked_chase_failures += 1
        self.path = []
        self.log(
            f"[COMBATE] CHASE bloqueado em {blocked} "
            f"(falhas={failures}, cooldown={cooldown:.1f}s)"
        )
        return failures

    def _combat_reposition_target(self, x, y, direction, observation):
        candidates = []
        preferred_dirs = self._left_right_dirs(direction)
        all_dirs = (*preferred_dirs, direction, TURN_LEFT_OF[TURN_LEFT_OF[direction]])
        seen_dirs = []
        for d in all_dirs:
            if d not in seen_dirs:
                seen_dirs.append(d)
        for dir_name in seen_dirs:
            vx, vy = DIR_VECTORS[dir_name]
            cell = (x + vx, y + vy)
            if not in_bounds(*cell) or cell in self._failed_reposition_cells:
                continue
            grid = self.world.grid[cell[0]][cell[1]]
            if grid == BLOCKED or grid in DANGEROUS:
                continue
            if not self.world.is_walkable(*cell, allow_unknown=False):
                continue
            same_axis = (
                (direction in ("north", "south") and dir_name in ("north", "south")) or
                (direction in ("east", "west") and dir_name in ("east", "west"))
            )
            score = 0.0
            score += 12.0 if dir_name in preferred_dirs else 0.0
            score += 8.0 if not same_axis else -3.0
            score += self.world.safe_exit_count(*cell) * 3.0
            score -= self.risk.cell_penalty(cell) * 0.08
            score -= self.world.visit_count.get(cell, 0) * 1.5
            candidates.append((score, cell))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _abandon_chase(self, reason):
        self.metrics.chases_abandoned += 1
        self._enemy_seen_at = 0.0
        self._enemy_seen_distance = None
        self.combat.hunt_turns = 0
        self.combat.hunt_reason = None
        self.path = []
        if self._restore_previous_plan():
            self._set_state("EXPLORE", f"{reason}; retomando plano anterior")
        else:
            self._set_state("EXPLORE", reason)

    def decide_state(self, x, y, energy, observation):
        pos = (x, y)
        now = time.time()

        if observation.damage:
            # Contra-ataque: com energia e agressividade suficientes (perfil-
            # dependente), combat.decide() retorna HUNT em vez de EVADE.
            # Energia baixa e perfil 'safe' continuam sempre evadindo.
            return self.combat.decide(energy, observation) or "EVADE"
        if energy <= LOW_ENERGY and self.farm.due_spots(("powerup",), energy):
            return "RECHARGE"
        if self.farm.has_collectable_item(pos, observation, energy, now):
            return "GRAB"
        if self.state == "REPOSITION" and self._reposition_steps_left > 0:
            return "REPOSITION"

        combat_state = None
        if self._enemy_fresh(observation):
            enemy_dist = observation.enemy_distance
            if not self._chase_in_cooldown(x, y, direction=self.world.orientation or "north",
                                           enemy_dist=enemy_dist):
                combat_state = self.combat.decide(energy, observation)
        if combat_state:
            return combat_state

        if self.state == "HUNT" and self.combat.hunt_turns > 0:
            return "HUNT"
        if self.state == "CAMP" and now - self.state_entered_at < 20:
            return "CAMP"
        return "EXPLORE"

    def act(self):
        view = self.sync()
        if view is None:
            self.sync_fails += 1
            if self.sync_fails % 20 == 1:
                self.log("[REDE] Servidor lento: aguardando dados frescos")
            return
        self.sync_fails = 0
        x, y, direction, pstate, score, energy, obs_tokens = view
        if x < 0:
            return

        observation = Observation.from_tokens(obs_tokens)
        self.metrics.decisions += 1
        self.tick_count += 1
        if observation.hit:
            self.metrics.hits += 1
        if observation.damage:
            self.metrics.damage += 1

        self.world.update_pose(x, y, direction)
        self._current_pos = (x, y)
        self._current_direction = direction
        self._update_enemy_memory(x, y, direction, observation)
        self._handle_blockage_and_motion(x, y, direction, observation)
        self.world.update_from_observation(x, y, observation)
        self.world.update_pose(x, y, direction)
        self.risk.update((x, y), observation)
        self.combat.update_after_observation(observation)
        self.farm.resolve_pending_grabs(score)
        self.farm.observe_respawn((x, y), observation)

        new_state = self.decide_state(x, y, energy, observation)
        self._set_state(new_state, f"pos=({x},{y}) energia={energy} obs={observation.as_list}")

        handler = {
            "EXPLORE": self.do_explore,
            "GRAB": self.do_grab,
            "RECHARGE": self.do_recharge,
            "ATTACK": self.do_attack,
            "CHASE": self.do_chase,
            "EVADE": self.do_evade,
            "HUNT": self.do_hunt,
            "FLEE": self.do_flee,
            "CAMP": self.do_camp,
            "REPOSITION": self.do_reposition,
        }[self.state]
        handler(x, y, direction, energy, observation, score)

    # ---------------- state handlers ----------------

    def do_grab(self, x, y, direction, energy, observation, score):
        self.ai.send_get_item()
        self._record_sent("t")
        self.last_action = "grab"
        self.last_target = None
        self._request_status_next()
        self._request_observation_next()
        self._clear_cached_light((x, y))
        self.farm.record_grab((x, y), score)
        self._set_state("EXPLORE", "item coletado")

    def _clear_cached_light(self, pos):
        """Remove o sinal de luz obsoleto do cache logo apos pegar o item:
        sem isso, observe_respawn le a mesma observacao (ainda com a luz de
        ANTES da coleta) nos proximos ticks ate a proxima resposta real do
        servidor, e confunde 'ainda nao atualizou' com 'respawnou em <1s',
        travando a estimativa de respawn no piso minimo."""
        if self._cached_view is None:
            return
        vx, vy, direction, pstate, score, energy, tokens = self._cached_view
        if (vx, vy) != pos:
            return
        stale_lights = {"bluelight", "redlight", "weaklight"}
        cleaned = [t for t in tokens if t.strip().lower() not in stale_lights]
        self._cached_view = (vx, vy, direction, pstate, score, energy, cleaned)

    def do_attack(self, x, y, direction, energy, observation, score):
        enemy_dist = observation.enemy_distance
        if not self.combat.should_attack(energy, observation):
            self.log("[COMBATE] Tiro bloqueado por risco/custo/linha de fogo")
            self._set_state("CHASE" if enemy_dist else "EXPLORE")
            return
        self.ai.send_shoot()
        self._record_sent("e")
        self.last_action = "shoot"
        self.last_target = None
        self.combat.record_shot()
        self._request_observation_next()
        self.log(f"[COMBATE] Atirar em enemy_dist={enemy_dist} energia={energy}")
        if self.combat.after_miss_limit(enemy_dist):
            self._set_state("HUNT", "limite de tiros sem hit")

    def do_chase(self, x, y, direction, energy, observation, score):
        self._chase_ticks += 1
        enemy_dist = observation.enemy_distance
        if not self._enemy_fresh(observation):
            self._abandon_chase("enemy expirado durante chase")
            return
        if self._chase_in_cooldown(x, y, direction, enemy_dist):
            self._set_state("REPOSITION", "cooldown de chase bloqueado")
            return
        if self._chase_ticks > self.config["max_chase_ticks"]:
            self._abandon_chase("limite economico de chase")
            return
        if self.combat.should_attack(energy, observation):
            self._set_state("ATTACK", "alvo em alcance")
            self.do_attack(x, y, direction, energy, observation, score)
            return
        if self.combat.can_chase_forward(x, y, direction):
            target = self._front_cell(x, y, direction)
            self.ai.send_forward()
            self._record_sent("w")
            self.last_action = "forward"
            self.last_target = target
            self._request_status_next()
            self._request_observation_next()
            self.log(f"[COMBATE] CHASE avancando para {target}")
            return
        failures = self._record_blocked_chase(x, y, direction, enemy_dist)
        if self.profile == "safe":
            self._set_state("EVADE", "chase bloqueado repetido")
            return
        if failures > self.config["max_blocked_chase_failures"]:
            self._abandon_chase("chase bloqueado repetido")
            return
        target = self._combat_reposition_target(x, y, direction, observation)
        if target is not None:
            self.path = [target]
            self.goal = target
            self.path_allows_flash = False
            self._reposition_steps_left = self.config["max_reposition_steps"]
            self._set_state("REPOSITION", "chase bloqueado; reposicionando")
            return
        self._abandon_chase("chase bloqueado sem flanqueamento seguro")

    def do_reposition(self, x, y, direction, energy, observation, score):
        if self._oscillating_chase_hunt():
            self.metrics.fsm_oscillations += 1
            self._abandon_chase("oscilacao CHASE/HUNT detectada")
            return
        if observation.damage:
            self._set_state("EVADE", "dano durante reposicionamento")
            return
        if self._reposition_steps_left <= 0:
            self._abandon_chase("reposicionamento sem progresso")
            return
        if not self.path:
            target = self._combat_reposition_target(x, y, direction, observation)
            if target is None:
                self._abandon_chase("sem reposicionamento seguro")
                return
            self.path = [target]
            self.goal = target
        target = self.path[0] if self.path else None
        self._follow_path(x, y, direction)
        self._reposition_steps_left -= 1
        if self.last_action in ("forward", "turn"):
            self.metrics.repositions += 1
        if target is not None and self.last_action == "wait":
            self._failed_reposition_cells.add(target)
        if not self.path:
            if self._enemy_fresh(observation) and self.combat.should_attack(energy, observation):
                self._set_state("ATTACK", "reposicionamento abriu tiro")
            elif self.profile == "aggressive" and self._enemy_fresh(observation):
                self._set_state("HUNT", "reposicionamento concluido")
            else:
                self._abandon_chase("reposicionamento concluido sem vantagem")

    def do_evade(self, x, y, direction, energy, observation, score):
        if not self.path:
            target = self.combat.best_evade_cell(x, y, direction, observation)
            if target is None:
                self._set_state("FLEE", "sem esquiva local")
                return
            self.path = [target]
            self.goal = target
            self.path_allows_flash = False
            self.log(f"[COMBATE] EVADE para {target}")
        self._follow_path(x, y, direction)
        if not self.path:
            self._set_state("EXPLORE", "esquiva concluida")

    def do_hunt(self, x, y, direction, energy, observation, score):
        if self._oscillating_chase_hunt():
            self.metrics.fsm_oscillations += 1
            self._abandon_chase("oscilacao CHASE/HUNT detectada")
            return
        if observation.enemy and self._enemy_fresh(observation):
            if self._chase_in_cooldown(x, y, direction, observation.enemy_distance):
                self._set_state("REPOSITION", "enemy visivel mas chase em cooldown")
                return
            self._set_state("ATTACK" if self.combat.should_attack(energy, observation) else "CHASE")
            return
        if observation.enemy and not self._enemy_fresh(observation):
            self._abandon_chase("enemy antigo ignorado no hunt")
            return
        if self.combat.hunt_turns <= 0:
            self.combat.hunt_turns = self.combat.hunt_turn_budget()
        self.ai.send_turn_right()
        self._record_sent("d")
        self.last_action = "turn"
        self.last_target = None
        self._request_status_next()
        self._request_observation_next()
        self.combat.hunt_turns -= 1
        self.log("[COMBATE] HUNT curto girando para procurar inimigo")
        if self.combat.hunt_turns <= 0:
            self.combat.hunt_reason = None
            self._set_state("EXPLORE", "scan concluido")

    def do_flee(self, x, y, direction, energy, observation, score):
        if not self.path:
            visited = [
                (cx, cy) for cx in range(59) for cy in range(34)
                if self.world.grid[cx][cy] == VISITED and (cx, cy) != (x, y)
            ]
            target, path = self.planner.flee_plan(
                (x, y), direction, visited,
                lambda cell: self.risk.flee_score(cell, (x, y), energy),
            )
            if not path:
                self.log("[COMBATE] Sem destino alcancavel para fuga")
                self._set_state("EXPLORE")
                return
            self.goal = target
            self.path = path
            self.path_allows_flash = False
            self.log(f"[COMBATE] FLEE para {target} ({len(path)} passos)")
        self._follow_path(x, y, direction)

    def do_recharge(self, x, y, direction, energy, observation, score):
        if not self.path:
            powerups = [p for p in self.farm.due_spots(("powerup",), energy=0) if p != (x, y)]
            goal, path = self.planner.nearest_reachable((x, y), powerups)
            if goal is None:
                self._set_state("EXPLORE", "sem powerup alcancavel")
                return
            refined = self.planner.a_star((x, y), goal, start_dir=direction, allow_unknown=True) or path
            self.goal = goal
            self.path = refined
            self.path_allows_flash = False
            self.log(f"[FARM] RECHARGE em {goal}")
        self._follow_path(x, y, direction)

    def do_camp(self, x, y, direction, energy, observation, score):
        if observation.has_item:
            self._set_state("GRAB", "respawn no camp")
            self.do_grab(x, y, direction, energy, observation, score)
            return
        now = time.time()
        pos = (x, y)
        if pos in self.world.item_spots and pos not in self.farm.last_taken:
            next_check = self.farm.next_check_at(pos, now)
            if next_check is not None and next_check > now:
                self._set_state("EXPLORE", "ponto sem luz atual; adiando nova checagem")
                return
        next_check = self.farm.next_check_at((x, y), now)
        if next_check is not None and now >= next_check - self.config["respawn_check_window"]:
            self._request_observation_next()
        if self.farm.has_better_reachable_plan((x, y), direction, energy):
            self._set_state("EXPLORE", "alvo melhor que camp")
            return
        if now - self.state_entered_at > 25 or self._looping():
            self._set_state("EXPLORE", "camp expirado")
            return
        self.last_action = "wait"
        self.last_target = None
        self.metrics.idle_without_command += 1
        if now - self._last_camp_log_at >= 2.0:
            wait = None if next_check is None else max(0.0, next_check - now)
            suffix = "" if wait is None else f" check_em={wait:.1f}s"
            self.log(f"[FARM] CAMP sem comando em ({x},{y}){suffix}")
            self._last_camp_log_at = now

    def do_explore(self, x, y, direction, energy, observation, score):
        if self.path and not observation.damage:
            self.metrics.plan_reused += 1
            self._follow_path(x, y, direction)
            return

        plan = self.farm.choose_plan((x, y), direction, energy, allow_flash=False)
        if plan is None:
            all_targets = list(self.world.item_spots) + self.world.frontier_cells()
            stuck = self._looping() or (self.idle_since and time.time() - self.idle_since > 8)
            if all_targets and self.risk.allow_teleport_fallback(False, stuck):
                goal, path = self.planner.teleport_fallback_plan(
                    (x, y), direction, all_targets,
                    lambda cell: -self.risk.cell_penalty(cell),
                )
                if path:
                    self.goal = goal
                    self.path = path
                    self.path_allows_flash = True
                    self.idle_since = None
                    self.log(f"[PLANO] Fallback com teleporte para {goal}")
            if not self.path:
                self._idle_or_probe(x, y)
        elif plan["label"] == "CAMP":
            self._set_state("CAMP", f"melhor utilidade em {plan['goal']}")
            self.do_camp(x, y, direction, energy, observation, score)
            return
        else:
            self.goal = plan["goal"]
            self.path = plan["path"]
            self.path_allows_flash = self.planner.path_allows_flash(self.path)
            self.idle_since = None
            self.log(f"[{plan['label']}] Rumo a {self.goal} ({len(self.path)} passos, u={plan['utility']:.0f})")
        self._follow_path(x, y, direction)

    def _idle_or_probe(self, x, y):
        now = time.time()
        if self.idle_since is None:
            self.idle_since = now
            self.last_action = "wait"
            self.last_target = None
            self.metrics.idle_without_command += 1
            self.log("[PLANO] Sem alvos alcancaveis: aguardando sem comando")
            return
        if now - self.idle_since > 12:
            unknown = [n for n in neighbors(x, y) if self.world.grid[n[0]][n[1]] == UNKNOWN]
            if unknown:
                best = min(unknown, key=lambda n: self.world.pit_risk(*n))
                if self.world.pit_risk(*best) <= 1:
                    self.path = [best]
                    self.path_allows_flash = False
                    self.idle_since = now
                    self.log(f"[PLANO] Impasse: sondando {best}")

    def _follow_path(self, x, y, direction):
        if not self.path:
            self.last_action = "wait"
            self.last_target = None
            self.metrics.idle_without_command += 1
            return
        if self.path[0] == (x, y):
            self.path.pop(0)
            if not self.path:
                self.last_action = "wait"
                self.last_target = None
                self.metrics.idle_without_command += 1
                return
        target = self.path[0]
        if abs(target[0] - x) + abs(target[1] - y) != 1:
            self.log("[PLANO] Caminho invalido: replanejando")
            self.path = []
            self.metrics.routes_cancelled += 1
            self.last_action = "wait"
            self.last_target = None
            self._request_status_next()
            self._request_observation_next()
            return
        cell = self.world.grid[target[0]][target[1]]
        flash_ok = self.path_allows_flash and cell == DANGER_FLASH
        if cell == BLOCKED or (cell in DANGEROUS and not flash_ok):
            self.log(f"[PLANO] Proxima celula {target} ficou insegura")
            self.path = []
            self.metrics.routes_cancelled += 1
            self.last_action = "wait"
            self.last_target = None
            return
        self.step_towards(x, y, direction, target)
