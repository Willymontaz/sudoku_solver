import copy
import time
from array import array
from typing import Tuple, List, Optional


# A performant class to represent a set of digits with 9 possible values
# It is used to represent all values used in a given list, row or block
# as well as possible solutions for a given cell during sudoku solving process
class Bitset:

    def __init__(self, initial_value=0):
        self.bitset = initial_value

    def add(self, d: int) -> None:
        self.bitset |= 1 << (d - 1)

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
        self.lines = [Bitset() for _ in range(9)]
        self.rows = [Bitset() for _ in range(9)]
        self.blocks = [Bitset() for _ in range(9)]

    def get_grid_idx(self, coords: Coords):
        return coords[0] * 9 + coords[1]

    def get_line(self, coords: Coords):
        return self.lines[coords[0]]

    def get_row(self, coords: Coords):
        return self.rows[coords[1]]

    def get_block(self, coords: Coords):
        block_id = int(coords[0] / 3) * 3 + int(coords[1] / 3)
        return self.blocks[block_id]

    def get_digit(self, coords: Coords):
        return self.grid[self.get_grid_idx(coords)]

    def update_value(self, coords: Coords, digit: int):
        assert digit != 0
        self.grid[self.get_grid_idx(coords)] = digit
        self.update_bitsets(coords, digit)

    def update_bitsets(self, coords: Coords, digit: int):
        self.get_line(coords).add(digit)
        self.get_row(coords).add(digit)
        self.get_block(coords).add(digit)

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


class Cell:

    def __init__(self, coords: Coords, possible_digits: Bitset):
        self.coords = coords
        self.possible_digits = possible_digits


class Solver:
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
                        combined_used_values = sudoku.get_line((x, y)).bitset | \
                                               sudoku.get_row((x, y)).bitset | \
                                               sudoku.get_block((x, y)).bitset
                        # The zeros are the values we are allowed to use, turn them to ones
                        possible_digits = ~combined_used_values & 0b111111111  # Invert and mask to 9 bits
                        resolvable_cells.append(Cell((x, y), Bitset(possible_digits)))

            unresolved_cells = []
            for cell in resolvable_cells:
                # If a cell has no option, this sudoku is not solvable
                if len(cell.possible_digits) == 0:
                    return None
                # If it has a single option, that's nice we make progress
                elif len(cell.possible_digits) == 1:
                    # Might have been inserted already in this loop, check that
                    digit = cell.possible_digits.get_digits()[0]
                    already_in_line = sudoku.get_line(cell.coords).contains(digit)
                    already_in_row = sudoku.get_row(cell.coords).contains(digit)
                    already_in_block = sudoku.get_block(cell.coords).contains(digit)
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
    def solve(sudoku: Sudoku) -> Sudoku:
        # We try to resolve everything we can directly resolve
        unresolved_cells = Solver.resolve_constrained_cells(sudoku)

        if unresolved_cells is None:
            raise 'Unsolvable sudoku'
        elif len(unresolved_cells) == 0:
            return sudoku
        else:
            # Use a stack to explore possibilities in depth-first search manner
            # We push Tuple[Coords, int, Sudoku] to the stack
            # Keeping a deep copy of the sudoku is important since resolve_constrained_cells
            # mutates its state. It would be a lot harder to keep track of all values discovered through constraints
            # and reset all of them
            exploration_stack = []
            Solver.update_exploration_stack(unresolved_cells, exploration_stack, sudoku)

            while len(exploration_stack) > 0:
                coords, digit, sudoku = exploration_stack.pop()
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
                    Solver.update_exploration_stack(unresolved_cells, exploration_stack, sudoku)

            # We are out of the loop, we explored everything and could not solve
            raise 'Unsolvable sudoku'

    @staticmethod
    def update_exploration_stack(unresolved_cells, exploration_stack, sudoku_before_exploration):
        # We explore by choosing the cell presenting the least possible branches
        candidate_cell = min(unresolved_cells, key=lambda cell: len(cell.possible_digits))
        for digit in candidate_cell.possible_digits.get_digits():
            # We keep track of the sudoku state at this step of the exploration
            exploration_stack.append((candidate_cell.coords, digit, copy.deepcopy(sudoku_before_exploration)))


if __name__ == '__main__':
    start_time = time.time()
    sudoku = Sudoku()
    # sudoku.load([
    #     [9,4,1,0,3,0,7,0,0],
    #     [0,0,5,0,0,8,6,0,0],
    #     [7,0,0,2,0,0,4,3,5],
    #     [0,1,0,0,5,0,0,4,3],
    #     [2,9,0,1,0,0,0,0,0],
    #     [8,5,0,7,4,0,9,0,0],
    #     [1,3,8,9,0,6,0,0,0],
    #     [0,0,0,0,1,0,0,8,2],
    #     [5,0,2,0,8,0,0,0,6],
    # ])
    # sudoku.load([
    #     [0,5,3,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,6,9],
    #     [0,0,0,7,2,0,0,0,0],
    #     [0,0,0,0,0,0,0,9,8],
    #     [4,0,0,6,0,0,0,0,7],
    #     [5,0,0,4,3,0,0,0,0],
    #     [0,0,2,5,0,6,0,0,0],
    #     [0,0,0,0,0,8,1,0,0],
    #     [0,8,9,0,0,7,0,0,4]
    # ])
    # sudoku.load([
    #     [0,8,0,0,0,4,0,5,0],
    #     [0,6,0,2,0,0,0,0,0],
    #     [5,0,2,0,7,0,1,0,0],
    #     [0,0,6,0,0,0,0,0,0],
    #     [2,0,1,9,0,0,0,4,0],
    #     [0,0,0,0,8,0,0,0,9],
    #     [0,0,0,0,0,3,7,0,0],
    #     [4,0,9,8,0,0,0,1,0],
    #     [0,5,0,0,0,0,0,0,0]
    # ])
    sudoku.load([
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
    solver = Solver()
    solved_sudoku = solver.solve(sudoku)
    print()
    print(solved_sudoku)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Duration: {duration} seconds")
