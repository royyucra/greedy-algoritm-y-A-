import sys

class Node():
    def __init__(self, state, parent, action, g=0, h=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g  # cost from start (used in A*)
        self.h = h  # heuristic cost to goal
        self.f = g + h  # total estimated cost

    def __lt__(self, other):
        return self.f < other.f


class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):
    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node


class GreedyFrontier(StackFrontier):
    """
    Frontier para Greedy Best-First Search.
    Siempre expande el nodo con menor valor heurístico h(n).
    """
    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        # Selecciona el nodo con menor h (distancia estimada al goal)
        best = min(self.frontier, key=lambda node: node.h)
        self.frontier.remove(best)
        return best


class AStarFrontier(StackFrontier):
    """
    Frontier para A* Search.
    Siempre expande el nodo con menor f(n) = g(n) + h(n).
    """
    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        # Selecciona el nodo con menor f = g + h
        best = min(self.frontier, key=lambda node: node.f)
        self.frontier.remove(best)
        return best


class Maze():

    def __init__(self, filename):
        with open(filename) as f:
            contents = f.read()

        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution = None

    def heuristic(self, state):
        """
        Heurística de distancia Manhattan entre el estado actual y el goal.
        h(n) = |row_actual - row_goal| + |col_actual - col_goal|
        Es admisible porque nunca sobreestima el costo real.
        """
        row, col = state
        goal_row, goal_col = self.goal
        return abs(row - goal_row) + abs(col - goal_col)

    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("█", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i, j) in solution:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()

    def neighbors(self, state):
        row, col = state
        candidates = [
            ("up",    (row - 1, col)),
            ("down",  (row + 1, col)),
            ("left",  (row, col - 1)),
            ("right", (row, col + 1))
        ]
        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result

    def solve_bfs(self):
        """BFS original - Breadth-First Search."""
        self.num_explored = 0
        start = Node(state=self.start, parent=None, action=None)
        frontier = QueueFrontier()
        frontier.add(start)
        self.explored = set()

        while True:
            if frontier.empty():
                raise Exception("no solution")
            node = frontier.remove()
            self.num_explored += 1

            if node.state == self.goal:
                actions, cells = [], []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            self.explored.add(node.state)
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action)
                    frontier.add(child)

    def solve_greedy(self):
        """
        Greedy Best-First Search.
        Expande siempre el nodo con menor h(n) (heurística al goal).
        No garantiza el camino óptimo, pero suele ser más rápido.
        """
        self.num_explored = 0
        h = self.heuristic(self.start)
        start = Node(state=self.start, parent=None, action=None, g=0, h=h)
        frontier = GreedyFrontier()
        frontier.add(start)
        self.explored = set()

        while True:
            if frontier.empty():
                raise Exception("no solution")

            # Selecciona nodo con menor h(n)
            node = frontier.remove()
            self.num_explored += 1

            if node.state == self.goal:
                actions, cells = [], []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            self.explored.add(node.state)
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    h = self.heuristic(state)
                    child = Node(state=state, parent=node, action=action, g=0, h=h)
                    frontier.add(child)

    def solve_astar(self):
        """
        A* Search.
        Expande el nodo con menor f(n) = g(n) + h(n).
        g(n) = costo real desde el inicio.
        h(n) = heurística (distancia Manhattan al goal).
        Garantiza el camino óptimo si h(n) es admisible.
        """
        self.num_explored = 0
        h = self.heuristic(self.start)
        start = Node(state=self.start, parent=None, action=None, g=0, h=h)
        frontier = AStarFrontier()
        frontier.add(start)
        self.explored = set()
        # Diccionario para rastrear el mejor g(n) conocido por estado
        best_g = {self.start: 0}

        while True:
            if frontier.empty():
                raise Exception("no solution")

            # Selecciona nodo con menor f(n) = g(n) + h(n)
            node = frontier.remove()
            self.num_explored += 1

            if node.state == self.goal:
                actions, cells = [], []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            self.explored.add(node.state)
            for action, state in self.neighbors(node.state):
                # Costo acumulado: cada paso tiene costo 1
                new_g = node.g + 1
                h = self.heuristic(state)

                # Solo agregar si no fue explorado o si encontramos un mejor camino
                if state not in self.explored:
                    if state not in best_g or new_g < best_g[state]:
                        best_g[state] = new_g
                        # Remover versión anterior del frontier si existe
                        self.frontier_remove_state(frontier, state)
                        child = Node(state=state, parent=node, action=action, g=new_g, h=h)
                        frontier.add(child)

    def frontier_remove_state(self, frontier, state):
        """Elimina un estado del frontier si existe (para A*)."""
        frontier.frontier = [n for n in frontier.frontier if n.state != state]

    def output_image(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw
        cell_size = 50
        cell_border = 2

        img = Image.new(
            "RGBA",
            (self.width * cell_size, self.height * cell_size),
            "black"
        )
        draw = ImageDraw.Draw(img)

        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    fill = (40, 40, 40)
                elif (i, j) == self.start:
                    fill = (255, 0, 0)
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)
                else:
                    fill = (237, 240, 252)

                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )

        img.save(filename)


# ─────────────────────────────────────────────
# Ejecución principal
# ─────────────────────────────────────────────
if len(sys.argv) < 2:
    sys.exit("Usage: python maze.py <maze.txt> [bfs|greedy|astar]")

maze_file = sys.argv[1]
algorithm = sys.argv[2].lower() if len(sys.argv) > 2 else "astar"

m = Maze(maze_file)
print("Maze:")
m.print()
print(f"Solving with: {algorithm.upper()}...")

if algorithm == "bfs":
    m.solve_bfs()
elif algorithm == "greedy":
    m.solve_greedy()
elif algorithm == "astar":
    m.solve_astar()
else:
    sys.exit("Algorithm must be: bfs, greedy, or astar")

print("States Explored:", m.num_explored)
print("Solution length:", len(m.solution[0]))
print("Solution:")
m.print()
m.output_image(f"maze_{algorithm}.png", show_explored=True)
print(f"Image saved as: maze_{algorithm}.png")
