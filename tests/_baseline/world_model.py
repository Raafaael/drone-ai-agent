"""
Modelo de mundo do drone: mapa 59x34 com conhecimento incremental,
inferencia logica estilo "Mundo de Wumpus" sobre os sensores
(breeze -> poco adjacente, flash -> teleporte adjacente) e busca
de caminhos com A*.

O agente nao conhece o mapa: tudo aqui e deduzido das observacoes.
"""

import heapq
from collections import deque

WIDTH = 59
HEIGHT = 34

# Estados de conhecimento de cada celula
UNKNOWN = "?"          # nunca visitada nem inferida
SAFE = "."             # inferida segura (sem poco/teleporte)
VISITED = "v"          # ja pisada (segura por definicao)
BLOCKED = "#"          # parede/obstaculo (impacto ao andar)
DANGER_PIT = "P"       # possivel poco
DANGER_FLASH = "T"     # possivel teleporte
DANGER_BOTH = "X"      # possivel poco e teleporte

DANGEROUS = {DANGER_PIT, DANGER_FLASH, DANGER_BOTH}

# vetores de direcao: north diminui y (convencao do GameServer)
DIR_VECTORS = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}


def in_bounds(x, y):
    return 0 <= x < WIDTH and 0 <= y < HEIGHT


def neighbors(x, y):
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny):
            yield nx, ny


class WorldModel:
    def __init__(self):
        self.grid = [[UNKNOWN] * HEIGHT for _ in range(WIDTH)]
        # celulas onde ja sentimos brisa/flash (para re-inferencia)
        self.breeze_cells = set()
        self.flash_cells = set()
        # celulas comprovadamente livres de poco / teleporte
        self.no_pit = set()
        self.no_flash = set()
        # resolucao logica: pocos/teleportes CONFIRMADOS por eliminacao
        # (brisa cujo unico vizinho nao-exonerado so pode ser o poco)
        self.confirmed_pits = set()
        self.confirmed_teleports = set()
        # itens vistos AGORA no chao: (x, y) -> "treasure" | "powerup" | "unknown"
        self.items = {}
        # memoria PERMANENTE de pontos de item (itens reaparecem -> farming)
        self.item_spots = {}

    # ---------------- atualizacao de conhecimento ----------------

    def mark_visited(self, x, y):
        if not in_bounds(x, y):
            return
        self.grid[x][y] = VISITED
        self.no_pit.add((x, y))
        self.no_flash.add((x, y))

    def mark_blocked(self, x, y):
        if in_bounds(x, y):
            self.grid[x][y] = BLOCKED

    def update_from_observation(self, x, y, obs):
        """Atualiza o conhecimento a partir das observacoes na celula (x, y)."""
        self.mark_visited(x, y)

        obs = [o.lower() for o in obs]  # servidor varia a grafia das luzes
        has_breeze = "breeze" in obs
        has_flash = "flash" in obs

        if has_breeze:
            self.breeze_cells.add((x, y))
        if has_flash:
            self.flash_cells.add((x, y))

        # Sem brisa => nenhum vizinho tem poco; sem flash => nenhum vizinho
        # tem teleporte (sensores de 1 passo manhattan, fora diagonais).
        for nx, ny in neighbors(x, y):
            if not has_breeze:
                self.no_pit.add((nx, ny))
            if not has_flash:
                self.no_flash.add((nx, ny))

        # itens na propria celula (e memoria permanente do ponto)
        kind = None
        if "bluelight" in obs:
            kind = "treasure"
        elif "redlight" in obs:
            kind = "powerup"
        elif "weaklight" in obs:
            kind = "unknown"
        if kind:
            self.items[(x, y)] = kind
            # nao rebaixa um ponto ja identificado para "unknown"
            if kind != "unknown" or (x, y) not in self.item_spots:
                self.item_spots[(x, y)] = kind

        self._reinfer()

    def _reinfer(self):
        """Reclassifica celulas desconhecidas com base no conhecimento atual.
        Aplica resolucao logica: toda brisa exige >= 1 poco vizinho; se sobrar
        um unico candidato, ele e poco confirmado (idem para flash)."""
        suspects_pit = set()
        for (bx, by) in self.breeze_cells:
            cands = [n for n in neighbors(bx, by)
                     if n not in self.no_pit and self.grid[n[0]][n[1]] != BLOCKED]
            if len(cands) == 1:
                self.confirmed_pits.add(cands[0])
            suspects_pit.update(cands)
        suspects_flash = set()
        for (fx, fy) in self.flash_cells:
            cands = [n for n in neighbors(fx, fy)
                     if n not in self.no_flash and self.grid[n[0]][n[1]] != BLOCKED]
            if len(cands) == 1:
                self.confirmed_teleports.add(cands[0])
            suspects_flash.update(cands)

        for x in range(WIDTH):
            for y in range(HEIGHT):
                cell = self.grid[x][y]
                if cell in (VISITED, BLOCKED):
                    continue
                p = (x, y) in suspects_pit
                f = (x, y) in suspects_flash
                if p and f:
                    self.grid[x][y] = DANGER_BOTH
                elif p:
                    self.grid[x][y] = DANGER_PIT
                elif f:
                    self.grid[x][y] = DANGER_FLASH
                elif (x, y) in self.no_pit and (x, y) in self.no_flash:
                    self.grid[x][y] = SAFE
                else:
                    self.grid[x][y] = UNKNOWN

    def consume_item(self, x, y):
        self.items.pop((x, y), None)

    def pit_risk(self, x, y):
        """Risco relativo de poco: numero de brisas adjacentes 'testemunhando'
        contra a celula (0 = nenhuma evidencia). Usado quando o agente esta
        encurralado e precisa escolher o menor risco."""
        if (x, y) in self.no_pit:
            return 0
        if (x, y) in self.confirmed_pits:
            return 99
        return sum(1 for (bx, by) in self.breeze_cells
                   if abs(bx - x) + abs(by - y) == 1)

    # ---------------- consultas ----------------

    def is_walkable(self, x, y, allow_unknown=False):
        if not in_bounds(x, y):
            return False
        cell = self.grid[x][y]
        if cell in (SAFE, VISITED):
            return True
        if allow_unknown and cell == UNKNOWN:
            return True
        return False

    def frontier_cells(self):
        """Celulas seguras/desconhecidas adjacentes a celulas visitadas:
        bons alvos de exploracao."""
        frontier = []
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.grid[x][y] not in (SAFE, UNKNOWN):
                    continue
                if any(self.grid[nx][ny] == VISITED for nx, ny in neighbors(x, y)):
                    frontier.append((x, y))
        return frontier

    # ---------------- buscas ----------------

    def nearest_reachable(self, start, goals, allow_flash=False):
        """BFS multi-alvo: retorna (alvo, caminho) do alvo mais proximo em
        'goals' alcancavel a partir de start, ou (None, None).

        Diferente de tentar A* alvo a alvo, uma unica varredura garante
        encontrar QUALQUER alvo alcancavel. A seguranca e a mesma do A*:
        celulas suspeitas de poco nunca entram; flash so com allow_flash.
        Celulas desconhecidas sao permitidas no plano porque a execucao
        valida celula a celula antes de pisar (a observacao da celula
        atual classifica os vizinhos antes do passo)."""
        goals = set(goals) - {start}
        if not goals:
            return None, None
        parent = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur in goals:
                path = []
                node = cur
                while node != start:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return cur, path
            for nxt in neighbors(*cur):
                if nxt in parent:
                    continue
                cell = self.grid[nxt[0]][nxt[1]]
                if cell == BLOCKED or cell in (DANGER_PIT, DANGER_BOTH):
                    continue
                if cell == DANGER_FLASH and not allow_flash:
                    continue
                parent[nxt] = cur
                queue.append(nxt)
        return None, None

    # ---------------- A* ----------------

    def a_star(self, start, goal, allow_unknown=False, start_dir=None,
               allow_flash=False):
        """A* no grid 'ciente de rotacao': cada giro de 90 graus custa 1 acao,
        entao caminhos retos sao preferidos. O no de busca e (celula, direcao).

        allow_flash: permite atravessar celulas suspeitas APENAS de teleporte
        (custo alto). Teleporte nao mata (-0 pontos); poco e fatal e nunca
        e atravessado. Use somente quando nao ha caminho seguro.

        Retorna lista de celulas do caminho (sem o start) ou None."""
        if start == goal:
            return []

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
                # o objetivo pode ser uma celula desconhecida (exploracao)
                walkable = self.is_walkable(*nxt, allow_unknown=allow_unknown) or nxt == goal
                if not walkable or self.grid[nxt[0]][nxt[1]] == BLOCKED:
                    continue
                # nunca atravessa celula suspeita de poco (fatal, -1000);
                # suspeita de teleporte so com allow_flash (ultimo recurso)
                cell = self.grid[nxt[0]][nxt[1]]
                if cell in (DANGER_PIT, DANGER_BOTH):
                    continue
                if cell == DANGER_FLASH and not allow_flash:
                    continue
                ndir = dir_of[(nxt[0] - pos[0], nxt[1] - pos[1])]
                # celulas desconhecidas custam mais (risco); flash-suspeitas
                # custam muito mais (teleporte aleatorio atrapalha o plano)
                if cell in (SAFE, VISITED):
                    step = 1
                elif cell == DANGER_FLASH:
                    step = 15
                else:
                    step = 3
                # custo de girar: 90 graus = 1 acao, 180 graus = 2 acoes
                if cdir is not None and ndir != cdir:
                    opposite = (DIR_VECTORS[cdir][0] == -DIR_VECTORS[ndir][0] and
                                DIR_VECTORS[cdir][1] == -DIR_VECTORS[ndir][1])
                    step += 2 if opposite else 1
                node = (nxt, ndir)
                ng = g + step
                if ng < g_cost.get(node, float("inf")):
                    g_cost[node] = ng
                    came_from[node] = current
                    heapq.heappush(open_heap, (ng + h(nxt, goal), ng, node))
        return None

    # ---------------- debug ----------------

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
