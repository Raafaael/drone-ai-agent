"""O que o drone sabe sobre o labirinto 59x34, construído aos poucos.

O agente não recebe o mapa pronto — cada célula começa `UNKNOWN` e só é
classificada a partir do que os sensores dizem na célula em que o drone está
agora. `breeze`/`flash` funcionam como no Mundo de Wumpus: indicam que HÁ um
poço/teleporte em algum vizinho, sem dizer qual — por eliminação (todo
vizinho sem evidência contrária é descartado), quando sobra um único
candidato ele vira poço/teleporte CONFIRMADO.

Esse conhecimento também alimenta o planejamento de rota (A*/BFS aqui
dentro): célula suspeita de poço nunca é considerada andável.
"""

import heapq
from collections import deque

from observations import Observation

WIDTH = 59
HEIGHT = 34

UNKNOWN = "?"
SAFE = "."
VISITED = "v"
BLOCKED = "#"
DANGER_PIT = "P"
DANGER_FLASH = "T"
DANGER_BOTH = "X"
DANGEROUS = {DANGER_PIT, DANGER_FLASH, DANGER_BOTH}

DIR_VECTORS = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}

TURN_LEFT_OF = {"north": "west", "west": "south", "south": "east", "east": "north"}
TURN_RIGHT_OF = {"north": "east", "east": "south", "south": "west", "west": "north"}


def in_bounds(x, y):
    return 0 <= x < WIDTH and 0 <= y < HEIGHT


def neighbors(x, y):
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny):
            yield nx, ny


def turn_cost(current_dir, target_dir):
    if current_dir is None or current_dir == target_dir:
        return 0
    if TURN_LEFT_OF[current_dir] == target_dir or TURN_RIGHT_OF[current_dir] == target_dir:
        return 1
    return 2


class WorldModel:
    """Mapa conhecido, memória de itens/ameaças e as buscas (A*/BFS) que
    andam sobre esse mapa. `version` sobe a cada mudança real na grade, e é
    o que o `Planner` usa para saber quando um caminho em cache ficou velho."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Esquece o mapa inteiro (nova partida: pode ser outro labirinto)."""
        self.grid = [[UNKNOWN] * HEIGHT for _ in range(WIDTH)]
        self.breeze_cells = set()
        self.flash_cells = set()
        self.no_pit = set()
        self.no_flash = set()
        self.confirmed_pits = set()
        self.confirmed_teleports = set()
        self.items = {}
        self.item_spots = {}
        self.visit_count = {}
        self.position = None
        self.orientation = "north"
        self.danger_cells = {}
        self.last_danger_cell = None
        self.threat_until = 0.0
        self.recent_positions = []
        self.version = 0

    def _changed(self):
        self.version += 1

    def update_pose(self, x, y, direction):
        self.position = (x, y)
        self.orientation = direction

    def mark_visited(self, x, y):
        if not in_bounds(x, y):
            return
        old = self.grid[x][y]
        self.grid[x][y] = VISITED
        self.no_pit.add((x, y))
        self.no_flash.add((x, y))
        self.visit_count[(x, y)] = self.visit_count.get((x, y), 0) + 1
        if not self.recent_positions or self.recent_positions[-1] != (x, y):
            self.recent_positions.append((x, y))
            if len(self.recent_positions) > 40:
                self.recent_positions.pop(0)
        if old != VISITED:
            self._changed()

    def mark_safe(self, x, y):
        if not in_bounds(x, y) or self.grid[x][y] in (VISITED, BLOCKED):
            return
        if self.grid[x][y] != SAFE:
            self.grid[x][y] = SAFE
            self._changed()

    def mark_blocked(self, x, y):
        if in_bounds(x, y) and self.grid[x][y] != BLOCKED:
            self.grid[x][y] = BLOCKED
            self.items.pop((x, y), None)
            self._changed()

    def front_cell(self, x=None, y=None, direction=None):
        if x is None or y is None:
            if self.position is None:
                return None
            x, y = self.position
        direction = direction or self.orientation
        dx, dy = DIR_VECTORS.get(direction, (0, 0))
        return x + dx, y + dy

    def update_from_observation(self, x, y, obs):
        """Incorpora o que os sensores disseram na célula (x, y): marca a
        célula como visitada, registra breeze/flash e o item no chão, e
        reclassifica o resto do mapa com o conhecimento novo."""
        observation = obs if isinstance(obs, Observation) else Observation.from_tokens(obs)
        self.update_pose(x, y, self.orientation)
        self.mark_visited(x, y)

        if observation.breeze:
            self.breeze_cells.add((x, y))
        if observation.flash:
            self.flash_cells.add((x, y))

        for nx, ny in neighbors(x, y):
            if not observation.breeze:
                self.no_pit.add((nx, ny))
            if not observation.flash:
                self.no_flash.add((nx, ny))

        if observation.light in ("treasure", "powerup", "unknown"):
            kind = observation.light
            self.items[(x, y)] = kind
            if kind != "unknown" or (x, y) not in self.item_spots:
                self.item_spots[(x, y)] = kind

        self._reinfer()

    def _reinfer(self):
        """Reclassifica toda célula ainda não visitada, à luz de todo
        breeze/flash já sentido: resolução por eliminação (ver docstring do
        módulo) confirma poço/teleporte quando só sobra um candidato."""
        old_version = self.version
        suspects_pit = set()
        for bx, by in self.breeze_cells:
            cands = [n for n in neighbors(bx, by)
                     if n not in self.no_pit and self.grid[n[0]][n[1]] != BLOCKED]
            if len(cands) == 1:
                self.confirmed_pits.add(cands[0])
            suspects_pit.update(cands)

        suspects_flash = set()
        for fx, fy in self.flash_cells:
            cands = [n for n in neighbors(fx, fy)
                     if n not in self.no_flash and self.grid[n[0]][n[1]] != BLOCKED]
            if len(cands) == 1:
                self.confirmed_teleports.add(cands[0])
            suspects_flash.update(cands)

        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.grid[x][y] in (VISITED, BLOCKED):
                    continue
                p = (x, y) in suspects_pit
                f = (x, y) in suspects_flash
                if p and f:
                    new = DANGER_BOTH
                elif p:
                    new = DANGER_PIT
                elif f:
                    new = DANGER_FLASH
                elif (x, y) in self.no_pit and (x, y) in self.no_flash:
                    new = SAFE
                else:
                    new = UNKNOWN
                if self.grid[x][y] != new:
                    self.grid[x][y] = new
                    self.version = old_version + 1

    def consume_item(self, x, y):
        if (x, y) in self.items:
            self.items.pop((x, y), None)
            self._changed()

    def pit_risk(self, x, y):
        """0 se comprovadamente seguro, 99 se poço confirmado, ou a
        contagem de brisas adjacentes testemunhando contra a célula (usado
        para comparar risco quando não há alternativa totalmente segura)."""
        if (x, y) in self.no_pit:
            return 0
        if (x, y) in self.confirmed_pits:
            return 99
        return sum(1 for bx, by in self.breeze_cells if abs(bx - x) + abs(by - y) == 1)

    def teleport_risk(self, x, y):
        if (x, y) in self.no_flash:
            return 0
        if (x, y) in self.confirmed_teleports:
            return 6
        return sum(1 for fx, fy in self.flash_cells if abs(fx - x) + abs(fy - y) == 1)

    def combined_risk(self, x, y):
        return self.pit_risk(x, y) * 100 + self.teleport_risk(x, y) * 10

    def safe_exit_count(self, x, y):
        """Quantos vizinhos parecem utilizáveis sem risco fatal — usado
        para não escolher becos sem saída como destino de fuga."""
        return sum(
            1 for nx, ny in neighbors(x, y)
            if self.grid[nx][ny] in (SAFE, VISITED, UNKNOWN, DANGER_FLASH)
            and self.grid[nx][ny] not in (DANGER_PIT, DANGER_BOTH, BLOCKED)
        )

    def is_walkable(self, x, y, allow_unknown=False, allow_flash=False):
        if not in_bounds(x, y):
            return False
        cell = self.grid[x][y]
        if cell in (SAFE, VISITED):
            return True
        if allow_unknown and cell == UNKNOWN:
            return True
        if allow_flash and cell == DANGER_FLASH:
            return True
        return False

    def can_enter(self, x, y, allow_unknown=False, allow_flash=False):
        if not in_bounds(x, y):
            return False
        cell = self.grid[x][y]
        if cell == BLOCKED or cell in (DANGER_PIT, DANGER_BOTH):
            return False
        if cell == DANGER_FLASH and not allow_flash:
            return False
        return self.is_walkable(x, y, allow_unknown=allow_unknown, allow_flash=allow_flash)

    def frontier_cells(self):
        """Células seguras/desconhecidas vizinhas de algo já visitado — os
        candidatos naturais para continuar explorando o mapa."""
        frontier = []
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.grid[x][y] not in (SAFE, UNKNOWN):
                    continue
                if any(self.grid[nx][ny] == VISITED for nx, ny in neighbors(x, y)):
                    frontier.append((x, y))
        return frontier

    def reachable_map(self, start, allow_flash=False, allow_unknown=True):
        """BFS a partir de `start`: distância e "de onde veio" para toda
        célula alcançável, numa única varredura (mais barato que rodar A*
        alvo por alvo quando é preciso ranquear vários candidatos)."""
        dist = {start: 0}
        parent = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nxt in neighbors(*cur):
                if nxt in dist:
                    continue
                if not self.can_enter(*nxt, allow_unknown=allow_unknown, allow_flash=allow_flash):
                    continue
                dist[nxt] = dist[cur] + 1
                parent[nxt] = cur
                queue.append(nxt)
        return dist, parent

    def path_from_parent(self, parent, goal):
        path = []
        node = goal
        while node is not None and parent.get(node) is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    def nearest_reachable(self, start, goals, allow_flash=False):
        """Entre `goals`, o mais próximo que dá para alcançar de fato
        (numa única varredura, em vez de testar alvo por alvo)."""
        goals = set(goals) - {start}
        if not goals:
            return None, None
        dist, parent = self.reachable_map(start, allow_flash=allow_flash, allow_unknown=True)
        reachable = [g for g in goals if g in dist]
        if not reachable:
            return None, None
        goal = min(reachable, key=lambda g: dist[g])
        return goal, self.path_from_parent(parent, goal)

    def farthest_reachable(self, start, goals, score_fn=None, allow_flash=False):
        """Como `nearest_reachable`, mas pega o alvo com maior `score_fn`
        (por padrão, o mais distante) — para fuga, por exemplo."""
        goals = set(goals) - {start}
        if not goals:
            return None, None
        dist, parent = self.reachable_map(start, allow_flash=allow_flash, allow_unknown=False)
        reachable = [g for g in goals if g in dist]
        if not reachable:
            return None, None
        if score_fn is None:
            def score_fn(cell):
                return dist[cell]
        goal = max(reachable, key=score_fn)
        return goal, self.path_from_parent(parent, goal)

    def a_star(self, start, goal, allow_unknown=False, start_dir=None,
               allow_flash=False, extra_cost=None):
        """Caminho mais curto até `goal`, contando giros como custo (cada
        90° vira +1 na busca) e nunca passando por célula suspeita de poço.
        `allow_flash` permite atravessar suspeita de teleporte (só como
        último recurso: teleporte não mata, poço sim). `extra_cost(cell)`,
        se passado, soma um custo de risco por célula do caminho."""
        if start == goal:
            return []
        if goal != start and not self.can_enter(*goal, allow_unknown=allow_unknown,
                                                allow_flash=allow_flash):
            return None

        def h(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        dir_of = {v: k for k, v in DIR_VECTORS.items()}
        start_node = (start, start_dir)
        open_heap = [(h(start, goal), 0, start_node)]
        came_from = {}
        g_cost = {start_node: 0}
        closed = set()

        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            pos, cdir = current
            if pos == goal:
                path = [pos]
                node = current
                while node in came_from:
                    node = came_from[node]
                    path.append(node[0])
                path.reverse()
                return path[1:]
            if current in closed:
                continue
            closed.add(current)
            for nxt in neighbors(*pos):
                if not self.can_enter(*nxt, allow_unknown=allow_unknown,
                                      allow_flash=allow_flash):
                    continue
                cell = self.grid[nxt[0]][nxt[1]]
                ndir = dir_of[(nxt[0] - pos[0], nxt[1] - pos[1])]
                if cell in (SAFE, VISITED):
                    step = 1.0
                elif cell == DANGER_FLASH:
                    step = 15.0
                else:
                    step = 3.0
                step += turn_cost(cdir, ndir)
                step += min(self.visit_count.get(nxt, 0), 8) * 0.35
                if extra_cost:
                    step += extra_cost(nxt)
                node = (nxt, ndir)
                ng = g + step
                if ng < g_cost.get(node, float("inf")):
                    g_cost[node] = ng
                    came_from[node] = current
                    heapq.heappush(open_heap, (ng + h(nxt, goal), ng, node))
        return None

    def render(self, px=None, py=None):
        rows = []
        for y in range(HEIGHT):
            row = []
            for x in range(WIDTH):
                if (x, y) == (px, py):
                    row.append("@")
                elif (x, y) in self.items:
                    row.append("$")
                else:
                    row.append(self.grid[x][y])
            rows.append("".join(row))
        return "\n".join(rows)
