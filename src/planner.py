"""Encontra caminhos no mapa: BFS para varreduras em massa e A* orientado
(cada giro de 90° custa uma ação a mais) para rotas precisas até um alvo.

O A* tem um cache simples por `world.version` — o mapa só muda quando uma
célula nova é classificada, então não há por que recalcular a mesma rota
tick após tick enquanto nada foi descoberto.
"""

from world_model import DANGER_FLASH, DIR_VECTORS, turn_cost

# Peso do risco por célula do caminho (poço/teleporte/revisita/ameaça
# recente). Menor que o peso usado na ESCOLHA do alvo (farm.py) porque aqui
# se acumula por CADA célula do trajeto, não uma vez só; ainda assim precisa
# ser alto o suficiente para preferir um desvio a atravessar uma zona onde
# o agente levou tiro ou viu um inimigo de perto há pouco tempo.
PATH_RISK_WEIGHT = 0.15


class Planner:
    """Fachada de busca de caminho usada pelo resto do agente: A* com
    cache, BFS multi-alvo e os planos de fuga/fallback de teleporte."""

    def __init__(self, world, risk=None):
        self.world = world
        self.risk = risk
        self._cache = {}

    def invalidate(self):
        """Limpa o cache de rotas (chamado sempre que o mapa muda)."""
        self._cache.clear()

    def _extra_cost(self, cell):
        return self.risk.cell_penalty(cell) * PATH_RISK_WEIGHT if self.risk else 0.0

    def reachable_map(self, start, allow_flash=False, allow_unknown=True):
        """BFS a partir de `start`: distância até TODAS as células
        alcançáveis numa única varredura, para ranquear vários alvos de
        uma vez sem repetir a busca por candidato."""
        return self.world.reachable_map(start, allow_flash=allow_flash,
                                        allow_unknown=allow_unknown)

    def path_from_parent(self, parent, goal):
        return self.world.path_from_parent(parent, goal)

    def a_star(self, start, goal, start_dir=None, allow_unknown=True,
               allow_flash=False):
        """Caminho mais curto até `goal`, ciente de rotação e de risco.
        Resultado fica em cache até o mapa mudar de versão."""
        key = (self.world.version, start, goal, start_dir, allow_unknown, allow_flash)
        if key not in self._cache:
            self._cache[key] = self.world.a_star(
                start, goal,
                allow_unknown=allow_unknown,
                start_dir=start_dir,
                allow_flash=allow_flash,
                extra_cost=self._extra_cost,
            )
        return None if self._cache[key] is None else list(self._cache[key])

    def nearest_reachable(self, start, goals, allow_flash=False):
        return self.world.nearest_reachable(start, goals, allow_flash=allow_flash)

    def best_scored_target(self, start, start_dir, targets, score_fn,
                           allow_flash=False, limit=60, max_path_len=None):
        """Entre `targets`, escolhe o de maior `score_fn(target)` que tenha
        rota alcançável, descontando o custo real de chegar até lá (passos
        + giros). Só considera os `limit` candidatos mais bem pontuados
        antes de rotear, para não gastar A* em alvos claramente ruins."""
        ranked = sorted(set(targets), key=score_fn, reverse=True)[:limit]
        best = None
        for target in ranked:
            path = self.a_star(start, target, start_dir=start_dir,
                               allow_unknown=True, allow_flash=allow_flash)
            if path is None:
                continue
            if max_path_len is not None and len(path) > max_path_len:
                continue
            score = score_fn(target) - self.path_action_cost(start, start_dir, path) * 1.4
            if best is None or score > best[0]:
                best = (score, target, path)
        return best

    def path_action_cost(self, start, start_dir, path):
        """Quantas ações (passos + giros) o caminho realmente custa."""
        if not path:
            return 0
        cost = 0
        cur = start
        cur_dir = start_dir
        dir_of = {v: k for k, v in DIR_VECTORS.items()}
        for nxt in path:
            delta = (nxt[0] - cur[0], nxt[1] - cur[1])
            nxt_dir = dir_of.get(delta)
            if nxt_dir is None:
                return 9999
            cost += 1 + turn_cost(cur_dir, nxt_dir)
            cur, cur_dir = nxt, nxt_dir
        return cost

    def flee_plan(self, start, start_dir, candidates, score_fn):
        """Escolhe o melhor destino de fuga entre `candidates` alcançáveis
        (por `score_fn`, tipicamente `risk.flee_score`) e traça a rota."""
        dist, parent = self.reachable_map(start, allow_flash=False, allow_unknown=False)
        reachable = [c for c in candidates if c in dist and c != start]
        if not reachable:
            return None, None
        for target in sorted(reachable, key=score_fn, reverse=True)[:30]:
            path = self.a_star(start, target, start_dir=start_dir,
                               allow_unknown=False, allow_flash=False)
            if path:
                return target, path
        target = max(reachable, key=score_fn)
        return target, self.path_from_parent(parent, target)

    def teleport_fallback_plan(self, start, start_dir, targets, score_fn):
        """Último recurso quando não há rota segura: permite atravessar
        células suspeitas apenas de teleporte (nunca de poço) para alcançar
        algum alvo."""
        flash_targets = set(targets)
        plan = self.best_scored_target(
            start, start_dir, flash_targets, score_fn,
            allow_flash=True, limit=80,
        )
        if plan:
            return plan[1], plan[2]
        goal, path = self.nearest_reachable(start, flash_targets, allow_flash=True)
        return goal, path

    def path_allows_flash(self, path):
        """True se o caminho passa por alguma célula suspeita de teleporte."""
        return any(self.world.grid[x][y] == DANGER_FLASH for x, y in path or [])
