"""Deterministic behavioral comparison for command economy.

The "before" agent below emulates the previous policy: every local tick calls
request_sync_pair(), therefore sending `q` and `o` even when waiting is free.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from strategy import DroneAgent


class SimAI:
    def __init__(self, obs=None, enemy_hit=False):
        self.x, self.y, self.direction = 5, 5, "north"
        self.score, self.energy = 0, 100
        self.observation = list(obs or [])
        self.actions = []
        self.pending_events = []
        self.enemy_hit = enemy_hit

    def _cost(self, command):
        self.score -= 1
        if command == "e":
            self.score -= 10

    def request_sync_pair(self, timeout=0.5):
        self.actions.extend(["q", "o"])
        self._cost("q")
        self._cost("o")
        obs = list(self.observation)
        obs.extend(self.consume_pending_events())
        return self.x, self.y, self.direction, "game", self.score, self.energy, obs

    def request_status_sync(self, timeout=0.5):
        self.actions.append("q")
        self._cost("q")
        return self.x, self.y, self.direction, "game", self.score, self.energy

    def request_observation_sync(self, timeout=0.5):
        self.actions.append("o")
        self._cost("o")
        obs = list(self.observation)
        obs.extend(self.consume_pending_events())
        return obs

    def consume_pending_events(self):
        events = list(self.pending_events)
        self.pending_events.clear()
        return events

    def send_forward(self):
        self.actions.append("w")
        self._cost("w")
        dx, dy = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}[self.direction]
        self.x += dx
        self.y += dy

    def send_backward(self):
        self.actions.append("s")
        self._cost("s")

    def send_turn_left(self):
        self.actions.append("a")
        self._cost("a")
        self.direction = {"north": "west", "west": "south", "south": "east", "east": "north"}[self.direction]

    def send_turn_right(self):
        self.actions.append("d")
        self._cost("d")
        self.direction = {"north": "east", "east": "south", "south": "west", "west": "north"}[self.direction]

    def send_get_item(self):
        self.actions.append("t")
        self._cost("t")
        if self.observation and any(o.lower() == "bluelight" for o in self.observation):
            self.score += 1000
        self.observation = []

    def send_shoot(self):
        self.actions.append("e")
        self._cost("e")
        if self.enemy_hit:
            self.pending_events.append("hit")


class AlwaysSyncAgent(DroneAgent):
    def sync(self):
        view = self.ai.request_sync_pair(timeout=0.5)
        self.metrics.sync_pairs += 1
        self.metrics.record_command("q")
        self.metrics.record_command("o")
        return view


class BuggyBlockedChaseAgent(DroneAgent):
    """Previous blocked-combat behavior: CHASE->HUNT->CHASE with no cooldown."""

    def _enemy_fresh(self, observation):
        return observation.enemy

    def _chase_in_cooldown(self, x, y, direction, enemy_dist):
        return False

    def do_chase(self, x, y, direction, energy, observation, score):
        if self.combat.should_attack(energy, observation):
            self._set_state("ATTACK", "alvo em alcance")
            return
        if self.combat.can_chase_forward(x, y, direction):
            target = self._front_cell(x, y, direction)
            self.ai.send_forward()
            self.metrics.record_command("w")
            self.last_action = "forward"
            self.last_target = target
            self._request_status_next()
            self._request_observation_next()
            return
        self.combat.hunt_reason = "chase"
        self.combat.hunt_turns = 2
        self._set_state("HUNT", "chase bloqueado")

    def do_hunt(self, x, y, direction, energy, observation, score):
        if observation.enemy:
            self._set_state("ATTACK" if self.combat.should_attack(energy, observation) else "CHASE")
            return
        self.ai.send_turn_right()
        self.metrics.record_command("d")
        self.last_action = "turn"


def setup_camp(agent, ai):
    agent._cached_view = (ai.x, ai.y, ai.direction, "game", ai.score, ai.energy, list(ai.observation))
    agent._last_status_at = time.time()
    agent._last_observation_at = time.time()
    agent._need_status = False
    agent._need_observation = False
    agent.world.update_from_observation(ai.x, ai.y, [])
    agent.world.item_spots[(ai.x, ai.y)] = "treasure"
    agent.farm.last_taken[(ai.x, ai.y)] = time.time()
    agent.farm.spot_respawn[(ai.x, ai.y)] = 30.0
    agent.farm.spot_next_check[(ai.x, ai.y)] = time.time() + 30.0
    agent.farm.has_better_reachable_plan = lambda *args, **kwargs: False
    agent.state = "CAMP"


def summarize(ai, agent):
    commands = ai.actions
    return {
        "Comandos totais": len(commands),
        "Observacoes o": commands.count("o"),
        "Consultas q": commands.count("q"),
        "Tiros": commands.count("e"),
        "Tiros sem acerto": commands.count("e") - agent.metrics.hits,
        "Rotas canceladas": agent.metrics.routes_cancelled,
        "Coletas": commands.count("t"),
        "Loops detectados": agent.metrics.loops_detected,
        "Acoes por minuto": agent.metrics.snapshot()["actions_per_min"],
        "Pontuacao simulada": ai.score,
    }


def run_stable(agent_cls):
    ai = SimAI([])
    agent = agent_cls(ai, log=lambda m: None, profile="score")
    setup_camp(agent, ai)
    ai.actions.clear()
    ai.score = 0
    for _ in range(60):
        agent.act()
    return summarize(ai, agent)


def run_steps_route(agent_cls):
    ai = SimAI(["steps"])
    agent = agent_cls(ai, log=lambda m: None, profile="score")
    agent._cached_view = (5, 5, "north", "game", 0, 100, ["steps"])
    agent._last_status_at = time.time()
    agent._last_observation_at = time.time()
    agent._need_status = False
    agent._need_observation = False
    agent.world.update_from_observation(5, 5, [])
    agent.world.update_from_observation(5, 4, [])
    agent.path = [(5, 4)]
    for _ in range(8):
        agent.act()
    return summarize(ai, agent)


def run_enemy_no_hit(agent_cls):
    ai = SimAI(["enemy#3"])
    agent = agent_cls(ai, log=lambda m: None, profile="aggressive")
    agent._cached_view = (5, 5, "north", "game", 0, 100, ["enemy#3"])
    agent._last_status_at = time.time()
    agent._last_observation_at = time.time()
    agent._need_status = False
    agent._need_observation = False
    for y in range(2, 6):
        agent.world.update_from_observation(5, y, [])
    agent.world.update_pose(5, 5, "north")
    for _ in range(12):
        agent.act()
    return summarize(ai, agent)


def setup_blocked_chase(agent, ai):
    agent._cached_view = (26, 23, "east", "game", 0, 100, ["enemy#6"])
    agent._last_status_at = time.time()
    agent._last_observation_at = time.time()
    agent._need_status = False
    agent._need_observation = False
    ai.x, ai.y, ai.direction = 26, 23, "east"
    ai.observation = ["enemy#6"]
    agent.world.update_from_observation(26, 23, [])
    agent.world.update_from_observation(26, 22, [])
    agent.world.update_from_observation(26, 24, [])
    agent.world.update_from_observation(25, 23, [])
    agent.world.mark_blocked(27, 23)
    agent.world.update_pose(26, 23, "east")


def run_blocked_chase(agent_cls):
    ai = SimAI(["enemy#6"])
    agent = agent_cls(ai, log=lambda m: None, profile="score")
    setup_blocked_chase(agent, ai)
    for _ in range(8):
        agent.act()
    snap = agent.metrics.snapshot(score=ai.score)
    transitions = snap["transitions"]
    return {
        "Transicoes CHASE/HUNT": transitions.get("CHASE->HUNT", 0) + transitions.get("HUNT->CHASE", 0),
        "Tempo na mesma posicao": 8 if (ai.x, ai.y) == (26, 23) else 0,
        "Bloqueios repetidos": snap["repeated_blocks"],
        "Reposicionamentos": snap["repositions"],
        "Perseguicoes abandonadas": snap["chases_abandoned"],
        "Plano anterior retomado": snap["previous_plan_restored"],
        "Comandos enviados": len(ai.actions),
    }


def compare():
    scenarios = {
        "A_estavel": run_stable,
        "C_steps_sem_inimigo": run_steps_route,
        "E_inimigo_sem_hit": run_enemy_no_hit,
    }
    rows = {}
    for name, fn in scenarios.items():
        before = fn(AlwaysSyncAgent)
        after = fn(DroneAgent)
        rows[name] = {"antes": before, "depois": after}
    return rows


def compare_blocked_chase():
    return {
        "antes": run_blocked_chase(BuggyBlockedChaseAgent),
        "depois": run_blocked_chase(DroneAgent),
    }


if __name__ == "__main__":
    results = compare()
    for scenario, data in results.items():
        print(f"\n[{scenario}]")
        for metric in data["antes"]:
            before = data["antes"][metric]
            after = data["depois"][metric]
            try:
                delta = after - before
            except TypeError:
                delta = ""
            print(f"{metric}: antes={before} depois={after} variacao={delta}")
    print("\n[CHASE_bloqueado]")
    blocked = compare_blocked_chase()
    for metric in blocked["antes"]:
        before = blocked["antes"][metric]
        after = blocked["depois"][metric]
        print(f"{metric}: antes={before} depois={after} variacao={after - before}")
