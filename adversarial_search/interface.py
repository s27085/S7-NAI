class InterfaceConstants:
    """
    Constants for the game interface display.
    Attributes:
        COLOR_BLUE (str): ANSI escape code for blue text.
        COLOR_RED (str): ANSI escape code for red text.
        COLOR_WHITE (str): ANSI escape code to reset text color to default.
    """
    COLOR_BLUE = '\033[34m'
    COLOR_RED = '\033[31m'
    COLOR_WHITE = '\033[0m'


def show_board(mankala_game, holes_per_row, alphabet):
    """
    Display the current game state to the console.
    Prints the board layout and player scores in a human-readable format.
    Args:
        mankala_game (Mankala): The current game instance.
        holes_per_row (int): The number of holes to display per row.
        alphabet (str): The alphabet string used for hole labeling.
    Side Effects:
        - Outputs to the console
    """

    player_two_alphabet = '  '.join(alphabet[holes_per_row:2 * holes_per_row].upper())

    print(player_two_alphabet + "     Score")

    row_output = ' '.join(['{:02d}'.format(i) for i in mankala_game.board[-1:-holes_per_row-1:-1]])
    print(row_output + f"  | Player 2: \033[31m" + str(mankala_game.players[1].score) + '\033[0m')

    row_output = ' '.join(['{:02d}'.format(i) for i in mankala_game.board[:holes_per_row]])
    print(row_output + f"  | Player 1: \033[34m" + str(mankala_game.players[0].score) + '\033[0m')

    player_one_alphabet = '  '.join(alphabet[:holes_per_row].upper())

    print(player_one_alphabet + "\n")

def show_draw():
    """
    Display a message indicating the game ended in a draw.
    Side Effects:
        - Outputs to the console
    """
    print(f"\n{InterfaceConstants.COLOR_BLUE}It's a draw!{InterfaceConstants.COLOR_WHITE}\n")


def show_winner(winner, score):
    """
    Display a message indicating which player won the game and their score.
    
    Args:
        winner (int): The index of the winning player (0 or 1).
        score (int): The final score of the winning player.

    Side Effects:
        - Outputs to the console
    """
    print(f"\n{InterfaceConstants.COLOR_BLUE if winner == 0 else InterfaceConstants.COLOR_RED}Player {winner + 1}{InterfaceConstants.COLOR_WHITE} wins with a score of {score}!\n")

def show_invalid_board_size():
    """
    Display an error message indicating the board size is invalid.
    Side Effects:
        - Outputs to the console
    """
    print(f"\n{InterfaceConstants.COLOR_RED}Board size has to be even. Aborting{InterfaceConstants.COLOR_WHITE}\n")

def show_maximum_board_size_exceeded():
    """
    Display an error message indicating the board size exceeds the maximum allowed.
    Side Effects:
        - Outputs to the console
    """
    print(f"\n{InterfaceConstants.COLOR_RED}Maximum board size exceeded. Aborting{InterfaceConstants.COLOR_WHITE}\n")