import sys
import time
import heapq
from collections import deque
from copy import deepcopy


class PuzzleState:
    def __init__(self, board, parent=None, move=None, depth=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.depth = depth
        self.empty_pos = self.find_empty()

    def __eq__(self, other):
        return self.board == other.board

    def __hash__(self):
        return hash(str(self.board))

    def __lt__(self, other):
        return False  # Needed for heapq

    def find_empty(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == 0:
                    return (i, j)
        return None

    def is_goal(self, goal_board):
        return self.board == goal_board

    def get_neighbors(self, move_order):
        neighbors = []
        i, j = self.empty_pos
        rows, cols = len(self.board), len(self.board[0])

        for move in move_order:
            if move == 'U' and i > 0:
                new_board = deepcopy(self.board)
                new_board[i][j], new_board[i - 1][j] = new_board[i - 1][j], new_board[i][j]
                neighbors.append(PuzzleState(new_board, self, 'U', self.depth + 1))
            elif move == 'D' and i < rows - 1:
                new_board = deepcopy(self.board)
                new_board[i][j], new_board[i + 1][j] = new_board[i + 1][j], new_board[i][j]
                neighbors.append(PuzzleState(new_board, self, 'D', self.depth + 1))
            elif move == 'L' and j > 0:
                new_board = deepcopy(self.board)
                new_board[i][j], new_board[i][j - 1] = new_board[i][j - 1], new_board[i][j]
                neighbors.append(PuzzleState(new_board, self, 'L', self.depth + 1))
            elif move == 'R' and j < cols - 1:
                new_board = deepcopy(self.board)
                new_board[i][j], new_board[i][j + 1] = new_board[i][j + 1], new_board[i][j]
                neighbors.append(PuzzleState(new_board, self, 'R', self.depth + 1))

        return neighbors

    def get_path(self):
        path = []
        current = self
        while current.parent is not None:
            path.append(current.move)
            current = current.parent
        return ''.join(reversed(path))


class PuzzleSolver:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.visited = 0
        self.processed = 0
        self.max_depth = 0

    def bfs(self, move_order):
        queue = deque([self.initial_state])
        visited = set()
        visited.add(str(self.initial_state.board))

        while queue:
            state = queue.popleft()
            self.processed += 1
            self.max_depth = max(self.max_depth, state.depth)

            if state.is_goal(self.goal_state.board):
                return state

            for neighbor in state.get_neighbors(move_order):
                if str(neighbor.board) not in visited:
                    visited.add(str(neighbor.board))
                    self.visited += 1
                    queue.append(neighbor)

        return None

    def dfs(self, move_order, max_depth=20):
        stack = [self.initial_state]
        visited = set()
        visited.add(str(self.initial_state.board))

        while stack:
            state = stack.pop()
            self.processed += 1
            self.max_depth = max(self.max_depth, state.depth)

            if state.is_goal(self.goal_state.board):
                return state

            if state.depth < max_depth:
                neighbors = state.get_neighbors(move_order)
                # Reverse to maintain order (stack is LIFO)
                for neighbor in reversed(neighbors):
                    if str(neighbor.board) not in visited:
                        visited.add(str(neighbor.board))
                        self.visited += 1
                        stack.append(neighbor)

        return None

    def a_star(self, heuristic):
        open_set = []
        heapq.heappush(open_set, (0, self.initial_state))
        g_score = {str(self.initial_state.board): 0}
        visited = set()

        while open_set:
            _, state = heapq.heappop(open_set)
            self.processed += 1
            self.max_depth = max(self.max_depth, state.depth)

            if state.is_goal(self.goal_state.board):
                return state

            if str(state.board) in visited:
                continue

            visited.add(str(state.board))
            self.visited += 1

            for neighbor in state.get_neighbors('URDL'):  # Order doesn't matter for A*
                tentative_g = g_score[str(state.board)] + 1
                if str(neighbor.board) not in g_score or tentative_g < g_score[str(neighbor.board)]:
                    g_score[str(neighbor.board)] = tentative_g
                    f_score = tentative_g + self.calculate_heuristic(neighbor, heuristic)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None

    def calculate_heuristic(self, state, heuristic):
        if heuristic == 'hamm':
            return self.hamming_distance(state)
        elif heuristic == 'manh':
            return self.manhattan_distance(state)
        else:
            return 0

    def hamming_distance(self, state):
        distance = 0
        for i in range(len(state.board)):
            for j in range(len(state.board[0])):
                if state.board[i][j] != 0 and state.board[i][j] != self.goal_state.board[i][j]:
                    distance += 1
        return distance

    def manhattan_distance(self, state):
        distance = 0
        size = len(state.board)
        goal_pos = {}

        # Create a map of value to position in goal state
        for i in range(size):
            for j in range(len(state.board[0])):
                goal_pos[self.goal_state.board[i][j]] = (i, j)

        for i in range(size):
            for j in range(len(state.board[0])):
                value = state.board[i][j]
                if value != 0:
                    goal_i, goal_j = goal_pos[value]
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance


def read_input_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        rows, cols = map(int, lines[0].strip().split())
        board = []
        for line in lines[1:rows + 1]:
            row = list(map(int, line.strip().split()))
            board.append(row)
        return board


def create_goal_board(initial_board):
    rows = len(initial_board)
    cols = len(initial_board[0])
    total = rows * cols
    goal = [[0 for _ in range(cols)] for _ in range(rows)]
    num = 1
    for i in range(rows):
        for j in range(cols):
            if num < total:
                goal[i][j] = num
                num += 1
    goal[rows - 1][cols - 1] = 0
    return goal


def write_solution_file(filename, solution):
    with open(filename, 'w') as f:
        if solution is None:
            f.write("-1\n")
        else:
            path = solution.get_path()
            f.write(f"{len(path)}\n")
            f.write(f"{path}\n")


def write_stats_file(filename, solution, visited, processed, max_depth, time_elapsed):
    with open(filename, 'w') as f:
        if solution is None:
            f.write("-1\n")
        else:
            f.write(f"{len(solution.get_path())}\n")
        f.write(f"{visited}\n")
        f.write(f"{processed}\n")
        f.write(f"{max_depth}\n")
        f.write(f"{time_elapsed:.3f}\n")


def main():
    if len(sys.argv) != 6:
        sys.stderr.write("Error: Invalid number of arguments\n")  # Write to stderr
        sys.exit(1)

    strategy = sys.argv[1]
    param = sys.argv[2]
    input_file = sys.argv[3]
    solution_file = sys.argv[4]
    stats_file = sys.argv[5]

    # Read initial board and create goal board
    initial_board = read_input_file(input_file)
    goal_board = create_goal_board(initial_board)

    initial_state = PuzzleState(initial_board)
    goal_state = PuzzleState(goal_board)

    solver = PuzzleSolver(initial_state, goal_state)

    start_time = time.time()

    if strategy == 'bfs':
        solution = solver.bfs(param)
    elif strategy == 'dfs':
        solution = solver.dfs(param)
    elif strategy == 'astr':
        solution = solver.a_star(param)
    else:
        print("Invalid strategy")
        sys.exit(1)

    time_elapsed = (time.time() - start_time) * 1000  # in milliseconds

    write_solution_file(solution_file, solution)
    write_stats_file(stats_file, solution, solver.visited, solver.processed, solver.max_depth, time_elapsed)


if __name__ == "__main__":
    main()
