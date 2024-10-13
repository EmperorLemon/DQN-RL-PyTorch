from board_module.board import Board
from board_module.logic import BoardLogic

import numpy as np


def test_game_logic():

    board = Board()

    # Test 1: Initial state
    assert board.grid.sum() in [
        4,
        6,
    ], "Initial board should have two tiles with sum 4 or 6"

    # Test 2: Move mechanics
    board.grid = np.array([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    BoardLogic.move(board, 0)  # Assuming 0 is 'up'
    assert board.grid[0, 0] == 4, "Tiles should merge when moving up"

    # Test 3: New tile generation
    empty_cells_before = np.count_nonzero(board.grid == 0)
    board.spawn_tile()
    empty_cells_after = np.count_nonzero(board.grid == 0)
    assert empty_cells_before - empty_cells_after == 1, "One new tile should be added"

    # Test 4:  over condition
    board.grid = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert BoardLogic.game_over(board), " should be over when no moves are possible"

    # Test 5: Score calculation
    board.grid = np.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    initial_score = board.get_score()
    BoardLogic.move(board, 3)
    assert (
        board.get_score() - initial_score == 4
    ), "Score should increase by 4 when two 2s merge"

    print("All game logic tests passed!")


test_game_logic()
