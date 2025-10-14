from easyAI import TwoPlayerGame
from interface import show_board

class Mankala(TwoPlayerGame):
    """
        A Mankala game implementation using the Oware variant rules.
        
        This class implements the traditional African board game Mankala (Oware variant)
        for two players. The game is played on a board with 12 holes (6 per player)
        and uses seeds as game pieces.
        
        Rules:
        - Each player starts with 4 seeds in each of their 6 holes
        - Players take turns picking up all seeds from one of their holes and
        distributing them one by one in subsequent holes (counter-clockwise)
        - If the last seed lands in an opponent's hole with 2 or 3 seeds (after
        placing), the player captures those seeds
        - The game ends when there are fewer than 7 seeds left on the board
        - The player with the most captured seeds wins
        
        Additional rules as defined in http://en.wikipedia.org/wiki/Oware:
        - The game ends when there are 6 seeds left in the game
        - Players must "feed" their opponent if possible
        
        Attributes:
            board (list): List of 12 integers representing seeds in each hole
            players (list): List of two Player objects
            current_player (int): Index of the current player (1 or 2)
        """
    def __init__(self, players):        
        """
        Initialize a new Mankala game.
        
        Sets up the game board with 4 seeds in each hole, initializes player
        attributes, and sets the starting player.
        
        Args:
            players (list): List of two Player objects (Human_Player or AI_Player)
        
        Returns:
            None
        """
        RED = '\033[31m'
        RESET = '\033[0m'

        STARTING_SCORE = 0
        WINNING_SCORE = 24
        STARTING_PLAYER = 1

        for numberOfPlayers, player in enumerate(players):
            player.score = STARTING_SCORE
            player.isstarved = False
            player.camp = numberOfPlayers
        self.players = players
        
        self.board = [4, 4, 4, 4, 4, 4,  
                      4, 4, 4, 4, 4, 4]
        
        if(len(self.board) % 2 != 0):
            print(f"\n{RED}Board size has to be even. Aborting{RESET}\n")
            return
                      
        self.current_player = STARTING_PLAYER
        self.winning_score = WINNING_SCORE
        

    def make_move(self, move):
        """
        Execute a move on the Mankala board.
        
        Implements the core game mechanics: distributing seeds from the chosen
        hole and capturing opponent's seeds when applicable. Handles special
        case when a player cannot move ("None" move).
        
        Args:
            move (str): Letter representing the hole to play ('a'-'l') or "None"
                       for no valid moves
        
        Returns:
            None
        
        Side Effects:
            - Modifies self.board by redistributing seeds
            - Updates player scores when seeds are captured
            - Sets player.isstarved flag if no moves available
        """
        def hole_in_enemy_row(hole):
            return (hole // 6) == self.opponent.camp
    
        def hole_has_proper_seeds(hole):
            return self.board[hole] in [2, 3]

        def distribute_seeds(starting_hole):
            stones_to_distribute = self, board
            for i in range(self.board[starting_hole]):
                hole_index = (hole_index + 1) % 12
                if hole_index == starting_hole:
                    continue
                self.board[hole_index] += 1

            self.board[starting_hole] = 0

        if move == "None":
            self.player.isstarved = True
            starting_hole_to_count_seeds = 6 * self.opponent.camp
            self.player.score += sum(self.board[starting_hole_to_count_seeds:starting_hole_to_count_seeds + 6])
            return

        starting_hole = 'abcdefghijkl'.index(move)

        hole_index = starting_hole
        for i in range(self.board[starting_hole]):
            hole_index = (hole_index + 1) % 12
            if hole_index == starting_hole:
                continue
            self.board[hole_index] += 1

        self.board[starting_hole] = 0
        

        while (hole_in_enemy_row(hole_index)
               and hole_has_proper_seeds(hole_index)):
            self.player.score += self.board[hole_index]
            self.board[hole_index] = 0
            hole_index = (hole_index - 1) % 12
    
    def possible_moves(self):
        """
        Check for possible moves for the current player.
        
        Returns a list of valid moves (holes) that the current player can play.
        If no valid moves are available, returns ['None'].
        If no hole has this many seeds, any non-empty hole can be played.
        Returns:
            list: List of valid moves as strings ('a'-'l') or ['None']
            if no moves are available

        Side Effects:
            - None
        Note: The current player is determined by self.current_player
        which is either 1 or 2, representing the two players.
        """
        def board_is_empty(board):
            return max(board) == 0
        
        first_row = self.board[:6]
        second_row = self.board[6:]
        
        row_values = first_row if self.current_player == 1 else second_row
        player_indices = range(6) if self.current_player == 1 else range(6, 12)

        if board_is_empty(row_values): return ['None']
        possible_moves = [i for i in player_indices if self.board[i] != 0]

        return ['abcdefghijkl'[u] for u in possible_moves]

    def show(self):
        """
        Display the current game state to the console.
        Prints the board layout and player scores in a human-readable format.
        Side Effects:
            - Outputs to the console
        """
        show_board(self)

    def lose(self):
        """
        Check if the current player has lost the game.
        A player loses if their opponent has captured more than 24 seeds.
        Returns:
            bool: True if the current player has lost, False otherwise
        """
        return self.opponent.score > self.winning_score

    def is_over(self):
        """
        Determine if the game has ended.
        The game ends if a player has lost, there are fewer than 7 seeds
        left on the board, or if the opponent is starved (cannot move).
        Returns:
            bool: True if the game is over, False otherwise
        """
        return ( self.lose() or
                  sum(self.board) < 7 or
                  self.opponent.isstarved )


if __name__ == "__main__":
    from easyAI import Human_Player, AI_Player, Negamax

    scoring = lambda game: game.player.score - game.opponent.score
    ai = Negamax(8, scoring)
    ai_2 = Negamax(6, scoring)
    game = Mankala([Human_Player(), AI_Player(ai_2)])

    game.play()

    if game.players[0].score > game.players[1].score:
        print("Player 1 wins with a score of %d." % game.players[0].score)
    elif game.players[0].score < game.players[1].score:
        print("Player 2 wins with a score of %d." % game.players[1].score)
    else:
        print("Draw.")