class PlaygroundConstants:
    """
    Constants for setting up the playground
    Attributes:
        AI_1_DEPTH (int): Search depth for AI player 1
        AI_2_DEPTH (int): Search depth for AI player 2
        IS_HUMAN_PLAYER_1 (bool): True if player 1 is human, False if AI.
        IS_HUMAN_PLAYER_2 (bool): True if player 2 is human, False if AI.
    """
    AI_1_DEPTH = 6
    AI_2_DEPTH = 2
    IS_HUMAN_PLAYER_1 = True
    IS_HUMAN_PLAYER_2 = False


class GameConstants:
    """
    Constants used in the Mankala game.

    Attributes:
        NUMBER_OF_HOLES_PER_PLAYER (int): Number of holes per player (6).
        NUMBER_OF_HOLES (int): Total number of holes on the board (12).
        STARTING_SEEDS_IN_HOLE (int): Initial number of seeds in each hole (4).
        HOLE_POINTS_TO_CAPTURE (list): List of hole points that can be captured (2, 3).

        STARTING_SCORE (int): Initial score for each player (0).
        WINNING_SCORE (int): Score required to win the game (24).
        STARTING_PLAYER (int): Index of the starting player (1).
        ALPHABET (str): String of lowercase letters used for hole labeling and moves.
    """ 
    NUMBER_OF_HOLES_PER_PLAYER = 6
    NUMBER_OF_HOLES = NUMBER_OF_HOLES_PER_PLAYER * 2
    STARTING_SEEDS_IN_HOLE = 4
    HOLE_POINTS_TO_CAPTURE = [2, 3]

    STARTING_SCORE = 0
    WINNING_SCORE = 24
    STARTING_PLAYER = 1
    ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

