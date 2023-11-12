import copy
import time
from abc import ABC, abstractmethod
from array import array
from typing import Tuple, List, Optional, Type


# A performant class to represent 9 sets of digits with 9 possible values for digits
# It is used to represent all values used in a given list, row or block
class ArrayBitset:

    def __init__(self):
        self.bitsets = array('i', [0] * 9)

    def get(self, idx: int):
        return self.bitsets[idx]

    def add(self, idx: int, d: int) -> None:
        self.bitsets[idx] |= 1 << (d - 1)

    def contains(self, idx: int, d: int) -> bool:
        return bool(self.bitsets[idx] & (1 << (d - 1)))


Coords = Tuple[int, int]


# Sudoku class stores a grid that is an array to remember the position and value of the digits in the sudoku
# Along we maintain lines, rows and blocks in the form of bitsets to efficiently compute possible solutions for
# a given cell
#
# For the purpose of the algorithm, the state can be mutated. However, when exploring branches of solutions
# one is supposed to deepcopy the sudoku instead of remembering and resetting the values mutated during the exploration
class Sudoku:

    def __init__(self):
        self.grid = array('i', [0] * 81)
        self.lines = ArrayBitset()
        self.rows = ArrayBitset()
        self.blocks = ArrayBitset()

    def get_grid_idx(self, coords: Coords):
        return coords[0] * 9 + coords[1]

    def get_line_bitset(self, coords: Coords):
        return self.lines.get(coords[0])

    def line_contains(self, coords: Coords, digit: int):
        return self.lines.contains(coords[0], digit)

    def line_add(self, coords: Coords, digit: int):
        self.lines.add(coords[0], digit)

    def get_row_bitset(self, coords: Coords):
        return self.rows.get(coords[1])

    def row_contains(self, coords: Coords, digit: int):
        return self.rows.contains(coords[1], digit)

    def row_add(self, coords: Coords, digit: int):
        self.rows.add(coords[1], digit)

    def get_block_id(self, coords: Coords):
        return int(coords[0] / 3) * 3 + int(coords[1] / 3)

    def get_block_bitset(self, coords: Coords):
        return self.blocks.get(self.get_block_id(coords))

    def block_contains(self, coords: Coords, digit: int):
        return self.blocks.contains(self.get_block_id(coords), digit)

    def block_add(self, coords: Coords, digit: int):
        self.blocks.add(self.get_block_id(coords), digit)

    def get_digit(self, coords: Coords):
        return self.grid[self.get_grid_idx(coords)]

    def update_value(self, coords: Coords, digit: int):
        assert digit != 0
        self.grid[self.get_grid_idx(coords)] = digit
        self.update_bitsets(coords, digit)

    def update_bitsets(self, coords: Coords, digit: int):
        self.line_add(coords, digit)
        self.row_add(coords, digit)
        self.block_add(coords, digit)

    # Load a soduko from a List[List[Int]]
    def load(self, lines):
        assert len(lines) == 9
        for i in range(9):
            line = lines[i]
            assert len(line) == 9
            for j in range(9):
                digit = line[j]
                self.grid[self.get_grid_idx((i, j))] = digit
                if digit != 0:
                    self.update_bitsets((i, j), digit)

    # Useful to print out a sudoku in terminal
    def __str__(self):
        s = ""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        s += str(self.grid[self.get_grid_idx((i * 3 + j, k * 3 + l))])
                        s += " "
                    s += " "
                s += "\n"
            s += "\n"

        return s


# A class used to represent possible digits for a cell as a single int
# Each bit set correspond to a possible solution
class PossibleDigits:

    def __init__(self, initial_value=0):
        self.bitset = initial_value

    # returns the number of digits in the set
    def __len__(self):
        count = 0
        b = self.bitset
        while b:
            b &= b - 1  # Clear the least significant set bit
            count += 1
        return count

    def get_digits(self) -> List[int]:
        digits = []
        for d in range(1, 10):
            if self.contains(d):
                digits.append(d)
        return digits

    def contains(self, d: int) -> bool:
        return bool(self.bitset & (1 << (d - 1)))
    

class Cell:

    def __init__(self, coords: Coords, possible_digits: PossibleDigits):
        self.coords = coords
        self.possible_digits = possible_digits


class Solver:
    # A special value used to initiate the solution exploration
    EXPLORATION_BOOTSTRAP = None

    class ExplorationTraversal(ABC):

        def __init__(self):
            self.exploration_tuples = [Solver.EXPLORATION_BOOTSTRAP]

        def has_more(self) -> bool:
            return len(self.exploration_tuples) > 0

        def add(self, exploration_tuple: Tuple[Coords, int, Sudoku]):
            self.exploration_tuples.append(exploration_tuple)

        @abstractmethod
        def next(self) -> Tuple[Coords, int, Sudoku]:
            pass

    # Depth-First-Search traversal kind
    class DFS(ExplorationTraversal):
        def next(self) -> Tuple[Coords, int, Sudoku]:
            return self.exploration_tuples.pop()

    # Breadth-First-Search traversal kind
    class BFS(ExplorationTraversal):
        def next(self) -> Tuple[Coords, int, Sudoku]:
            return self.exploration_tuples.pop(0)

    # Resolves cells that can be resolved by pure constraint
    # returns the unresolved cells
    # Fails if the sudoku cannot be solved
    #
    # Warning: the sudoku is mutated by this method and is supposed to be reused for further work
    @staticmethod
    def resolve_constrained_cells(sudoku: Sudoku) -> Optional[List[Cell]]:
        previous_unresolved_count = 81  # We consider all cells unresolved at first
        while True:
            resolvable_cells = []
            for x in range(9):
                for y in range(9):
                    digit = sudoku.get_digit((x, y))
                    if digit == 0:  # We don't resolve cells having already a value
                        # Compute possible values thanks to line, rows and block bitsets
                        combined_used_values = sudoku.get_line_bitset((x, y)) | \
                                               sudoku.get_row_bitset((x, y)) | \
                                               sudoku.get_block_bitset((x, y))
                        # The zeros are the values we are allowed to use, turn them to ones
                        possible_digits = ~combined_used_values & 0b111111111  # Invert and mask to 9 bits
                        resolvable_cells.append(Cell((x, y), PossibleDigits(possible_digits)))

            unresolved_cells = []
            for cell in resolvable_cells:
                # If a cell has no option, this sudoku is not solvable
                if len(cell.possible_digits) == 0:
                    return None
                # If it has a single option, that's nice we make progress
                elif len(cell.possible_digits) == 1:
                    # Might have been inserted already in this loop, check that
                    digit = cell.possible_digits.get_digits()[0]
                    already_in_line = sudoku.line_contains(cell.coords, digit)
                    already_in_row = sudoku.row_contains(cell.coords, digit)
                    already_in_block = sudoku.block_contains(cell.coords, digit)
                    if not already_in_line and not already_in_row and not already_in_block:
                        sudoku.update_value(cell.coords, digit)
                    else:
                        # Oops, that was a situation where constraints led to put the same
                        # digit twice either in line, row or block
                        # This sudoku is unsolvable
                        return None
                else:
                    unresolved_cells.append(cell)

            # Verify some possible loop exits
            if len(unresolved_cells) == 0:
                # The sudoku is resolved, return an empty list
                return unresolved_cells
            elif len(unresolved_cells) == previous_unresolved_count:
                # There are remaining unresolved cells, and we made no progress with this loop
                # Return the unresolved cells for exploratory work
                return unresolved_cells
            else:
                # Maybe thanks to the resolved cells we can now resolve other cells
                # Keep track of the previous count and check later if we make progress
                previous_unresolved_count = len(unresolved_cells)

    # Solves a sudoku, possibly through exploration
    # Returns the sudoku solution (the provided sudoku is mutated but the
    # resulting mutation is not necessarily the solution)
    @staticmethod
    def solve(sudoku: Sudoku, traversal_kind: Type[ExplorationTraversal] = DFS) -> Optional[Sudoku]:
        # We push Tuple[Coords, int, Sudoku] to the exploration_traversal
        # Keeping a deep copy of the sudoku is important since resolve_constrained_cells
        # mutates its state. It would be a lot harder to keep track of all values discovered through constraints
        # and reset all of them

        # Init the exploration
        exploration_traversal = traversal_kind()

        while exploration_traversal.has_more():
            exploration_tuple = exploration_traversal.next()
            if exploration_tuple != Solver.EXPLORATION_BOOTSTRAP:
                coords, digit, sudoku = exploration_tuple
                sudoku.update_value(coords, digit)

            unresolved_cells = Solver.resolve_constrained_cells(sudoku)
            if unresolved_cells is None:
                # This exploration led to dead end, explore another branch
                continue
            elif len(unresolved_cells) == 0:
                # This sudoku is solved, return it
                return sudoku
            else:
                # Otherwise, we need to continue exploration
                Solver.update_exploration_traversal(unresolved_cells, exploration_traversal, sudoku)

        # We are out of the loop, we explored everything and could not solve
        return None

    @staticmethod
    def update_exploration_traversal(unresolved_cells, exploration_traversal: ExplorationTraversal,
                                     sudoku_before_exploration):
        # We explore by choosing the cell presenting the least possible branches
        candidate_cell = min(unresolved_cells, key=lambda cell: len(cell.possible_digits))
        for digit in candidate_cell.possible_digits.get_digits():
            # We keep track of the sudoku state at this step of the exploration
            exploration_traversal.add((candidate_cell.coords, digit, copy.deepcopy(sudoku_before_exploration)))


def verify_solution_matches_problem(problem: List[List[int]], solution: Sudoku):
    for i in range(9):
        for j in range(9):
            problem_digit = problem[i][j]
            solution_digit = solution.get_digit((i, j))
            if problem_digit != 0 and solution_digit != problem_digit:
                raise RuntimeError(
                    f"The solution does not match the problem at cell {(i, j)}. "
                    f"Expected {problem_digit}, got {solution_digit}"
                )


def verify_sudoku_correctness(solution: Sudoku):
    import itertools
    lines = {}
    rows = {}
    blocks = {}
    for i in range(9):
        for j in range(9):
            digit = solution.get_digit((i, j))
            lines.setdefault(i, []).append(digit)
            rows.setdefault(j, []).append(digit)
            blocks.setdefault(int(i / 3) * 3 + int(j / 3), []).append(digit)
    for digit_list in itertools.chain(lines.values(), rows.values(), blocks.values()):
        if len(digit_list) != len(set(digit_list)):
            raise RuntimeError("The solution is invalid")


def run_and_verify_with_traversal(problem, traversal_kind):
    sudoku = Sudoku()
    sudoku.load(problem)
    solver = Solver()
    start_time = time.time()
    solved_sudoku = solver.solve(sudoku, traversal_kind=traversal_kind)
    end_time = time.time()
    print(solved_sudoku)
    verify_solution_matches_problem(problem, solved_sudoku)
    verify_sudoku_correctness(solved_sudoku)
    print()
    duration = end_time - start_time
    print(f"Tests duration using {traversal_kind}: {duration} seconds")
    print()


def run_and_verify(problem):
    run_and_verify_with_traversal(problem, Solver.DFS)
    run_and_verify_with_traversal(problem, Solver.BFS)


if __name__ == '__main__':
    print("Problem easy")
    run_and_verify([
        [9, 4, 1, 0, 3, 0, 7, 0, 0],
        [0, 0, 5, 0, 0, 8, 6, 0, 0],
        [7, 0, 0, 2, 0, 0, 4, 3, 5],
        [0, 1, 0, 0, 5, 0, 0, 4, 3],
        [2, 9, 0, 1, 0, 0, 0, 0, 0],
        [8, 5, 0, 7, 4, 0, 9, 0, 0],
        [1, 3, 8, 9, 0, 6, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 8, 2],
        [5, 0, 2, 0, 8, 0, 0, 0, 6],
    ])
    print("Problem medium")
    run_and_verify([
        [0, 5, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 6, 9],
        [0, 0, 0, 7, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 9, 8],
        [4, 0, 0, 6, 0, 0, 0, 0, 7],
        [5, 0, 0, 4, 3, 0, 0, 0, 0],
        [0, 0, 2, 5, 0, 6, 0, 0, 0],
        [0, 0, 0, 0, 0, 8, 1, 0, 0],
        [0, 8, 9, 0, 0, 7, 0, 0, 4]
    ])
    print("Problem hard")
    run_and_verify([
        [0, 8, 0, 0, 0, 4, 0, 5, 0],
        [0, 6, 0, 2, 0, 0, 0, 0, 0],
        [5, 0, 2, 0, 7, 0, 1, 0, 0],
        [0, 0, 6, 0, 0, 0, 0, 0, 0],
        [2, 0, 1, 9, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 9],
        [0, 0, 0, 0, 0, 3, 7, 0, 0],
        [4, 0, 9, 8, 0, 0, 0, 1, 0],
        [0, 5, 0, 0, 0, 0, 0, 0, 0]
    ])
    print("Problem master class")
    run_and_verify([
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0]
    ])
