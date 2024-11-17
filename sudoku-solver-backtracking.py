import numpy as np
from typing import List, Tuple, Optional
import time

class Sudoku:
    def __init__(self, puzzle: List[List[int]]):
        self.original = np.array(puzzle)
        self.puzzle = np.array(puzzle)
        self.fixed_positions = self.original != 0
        
    def is_valid(self) -> bool:
        """Check if the current state is valid and complete."""
        # Check for zeros
        if 0 in self.puzzle:
            return False
            
        # Check rows
        for row in self.puzzle:
            if not self._is_valid_unit(row):
                return False
        
        # Check columns
        for col in self.puzzle.T:
            if not self._is_valid_unit(col):
                return False
        
        # Check 3x3 boxes
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                box = self.puzzle[i:i+3, j:j+3].flatten()
                if not self._is_valid_unit(box):
                    return False
        
        return True
    
    def _is_valid_unit(self, unit) -> bool:
        """Check if a row/column/box contains unique numbers 1-9."""
        unit = [x for x in unit if x != 0]
        return len(unit) == len(set(unit))
    
    def get_fitness(self) -> float:
        """Calculate fitness score (higher is better)."""
        score = 0
        
        # Heavily penalize remaining zeros
        zeros_penalty = -100 * np.count_nonzero(self.puzzle == 0)
        score += zeros_penalty
        
        # Score rows
        for row in self.puzzle:
            non_zero = row[row != 0]
            score += len(set(non_zero)) - (9 - len(non_zero))
        
        # Score columns
        for col in self.puzzle.T:
            non_zero = col[col != 0]
            score += len(set(non_zero)) - (9 - len(non_zero))
        
        # Score boxes
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                box = self.puzzle[i:i+3, j:j+3].flatten()
                non_zero = box[box != 0]
                score += len(set(non_zero)) - (9 - len(non_zero))
        
        return score
    
    def display(self):
        """Print the Sudoku grid."""
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("-" * 21)
            
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print("|", end=" ")
                print(self.puzzle[i][j], end=" ")
            print()
    
    def __eq__(self, other):
        """Check if two Sudoku puzzles have the same solution."""
        if not isinstance(other, Sudoku):
            return False
        return np.array_equal(self.puzzle, other.puzzle)
    
    def __hash__(self):
        """Hash function for Sudoku puzzle."""
        return hash(self.puzzle.tobytes())

class SudokuSolver:
    @staticmethod
    def find_empty(puzzle: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find an empty cell in the puzzle."""
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] == 0:
                    return (i, j)
        return None
    
    @staticmethod
    def is_valid_move(puzzle: np.ndarray, pos: Tuple[int, int], num: int) -> bool:
        """Check if placing a number at the given position is valid."""
        # Check row
        if num in puzzle[pos[0]]:
            return False
        
        # Check column
        if num in puzzle[:, pos[1]]:
            return False
        
        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if puzzle[i][j] == num:
                    return False
        
        return True
    
    def solve_recursive(self, puzzle: np.ndarray) -> bool:
        """Solve Sudoku using backtracking."""
        empty = self.find_empty(puzzle)
        if not empty:
            return True
        
        row, col = empty
        for num in range(1, 10):
            if self.is_valid_move(puzzle, (row, col), num):
                puzzle[row][col] = num
                
                if self.solve_recursive(puzzle):
                    return True
                
                puzzle[row][col] = 0
        
        return False
    
    def find_all_solutions(self, puzzle: List[List[int]], max_solutions: int = 10) -> List[Sudoku]:
        """Find multiple solutions for a Sudoku puzzle."""
        solutions = set()
        puzzle_array = np.array(puzzle)
        
        def solve_for_multiple(current_puzzle: np.ndarray) -> None:
            if len(solutions) >= max_solutions:
                return
            
            empty = self.find_empty(current_puzzle)
            if not empty:
                # Found a solution
                solution = Sudoku(current_puzzle.copy())
                solutions.add(solution)
                return
            
            row, col = empty
            for num in range(1, 10):
                if self.is_valid_move(current_puzzle, (row, col), num):
                    current_puzzle[row][col] = num
                    solve_for_multiple(current_puzzle)
                    current_puzzle[row][col] = 0
        
        solve_for_multiple(puzzle_array)
        return list(solutions)
    
    def has_unique_solution(self, puzzle: List[List[int]]) -> bool:
        """Check if the puzzle has exactly one solution."""
        solutions = self.find_all_solutions(puzzle, max_solutions=2)
        return len(solutions) == 1

if __name__ == "__main__":
    """Extremely hard puzzle"""
    # puzzle = [
    #     [4,0,0,0,0,0,8,3,0],
    #     [0,0,0,0,0,0,0,4,0],
    #     [0,0,0,8,2,0,0,0,0],
    #     [0,6,0,0,0,0,0,0,0],
    #     [0,8,0,0,7,0,0,0,0],
    #     [0,0,0,2,9,0,5,0,1],
    #     [5,0,1,0,0,0,9,0,4],
    #     [0,0,9,3,0,0,0,0,7],
    #     [0,0,0,0,5,9,0,0,3]
    # ]
    """Puzzle with multiple solutions"""
    puzzle = [
        [2,9,5,7,4,3,8,6,1],
        [4,3,1,8,6,5,9,0,0],
        [8,7,6,1,9,2,5,4,3],
        [3,8,7,4,5,9,2,1,6],
        [6,1,2,3,8,7,4,9,5],
        [5,4,9,2,1,6,7,3,8],
        [7,6,3,5,3,4,1,8,9],
        [9,2,8,6,7,1,3,5,4],
        [1,5,4,9,3,8,6,0,0]
    ]
    
    print("Original puzzle:")
    sudoku = Sudoku(puzzle)
    sudoku.display()
    print(f"Current fitness: {sudoku.get_fitness()}")
    
    # Check if puzzle has unique solution
    solver = SudokuSolver()
    has_unique = solver.has_unique_solution(puzzle)
    print(f"\nPuzzle has unique solution: {has_unique}")
    
    print("\n=== Traditional Solver ===")
    start_time = time.time()
    traditional_solutions = solver.find_all_solutions(puzzle, max_solutions=5)
    trad_time = time.time() - start_time
    
    print(f"\nFound {len(traditional_solutions)} solutions using traditional solver")
    print(f"Time taken: {trad_time:.2f} seconds")
    
    for i, solution in enumerate(traditional_solutions, 1):
        print(f"\nTraditional Solution {i}:")
        solution.display()
        print(f"Fitness: {solution.get_fitness()}")
        