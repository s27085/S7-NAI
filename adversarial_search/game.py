from easyAI import TwoPlayerGame
from interface import *
from cli_parser import handle_arguments
from constants import *




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
            board (list): List of 12 (GameConstants.NUMBER_OF_HOLES) integers representing seeds in each hole
            players (list): List of two Player objects
            current_player (int): Index of the current player (1 or 2)
        """
    def __init__(self, players):        
        """
        Initialize a new Mankala game.
        
        Sets up the game board with 4 (GameConstants.STARTING_SEEDS_IN_HOLE) seeds in each hole, initializes player
        attributes, and sets the starting player.
        
        Args:
            players (list): List of two Player objects (Human_Player or AI_Player)
        
        Returns:
            None
        """

        self.board = GameConstants.NUMBER_OF_HOLES * [GameConstants.STARTING_SEEDS_IN_HOLE]

        for number_of_players, player in enumerate(players):
            player.isstarved = False
            player.camp = number_of_players
            player.score = GameConstants.STARTING_SCORE
        self.players = players
        
        if(GameConstants.NUMBER_OF_HOLES > len(GameConstants.ALPHABET)):
            show_maximum_board_size_exceeded()
            raise ValueError(f"Board size ({GameConstants.NUMBER_OF_HOLES}) exceeds alphabet length ({len(GameConstants.ALPHABET)})")

        self.current_player = GameConstants.STARTING_PLAYER
        self.winning_score = GameConstants.WINNING_SCORE


    def possible_moves(self):
        """
        Check for possible moves for the current player.
        
        Returns a list of valid moves (holes) that the current player can play.
        If no valid moves are available, returns ['None'].
        Any non-empty hole can be played.
        Returns:
            list: List of valid moves as strings ('a'-'l') or ['None']
            if no moves are available

        Side Effects:
            - None
        Note: The current player is determined by self.current_player
        which is either 1 or 2, representing the two players.
        """
        def holes_are_empty(board):
            return max(board) == 0
        
        if self.current_player == 1:
            row = self.board[:GameConstants.NUMBER_OF_HOLES_PER_PLAYER]
            player_indices = range(GameConstants.NUMBER_OF_HOLES_PER_PLAYER)
        else:
            row = self.board[GameConstants.NUMBER_OF_HOLES_PER_PLAYER:]
            player_indices = range(GameConstants.NUMBER_OF_HOLES_PER_PLAYER, GameConstants.NUMBER_OF_HOLES)

        if holes_are_empty(row): return ['None']
        possible_moves = [i for i in player_indices if self.board[i] != 0]

        return [GameConstants.ALPHABET[u] for u in possible_moves]

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
            return (hole // GameConstants.NUMBER_OF_HOLES_PER_PLAYER) == self.opponent.camp
    
        def hole_has_required_seeds(hole):
            return self.board[hole] in GameConstants.HOLE_POINTS_TO_CAPTURE

        def distribute_seeds(starting_hole):
            seeds_to_sow = self.board[starting_hole]
            self.board[starting_hole] = 0
            hole_index = starting_hole

            for _ in range(seeds_to_sow):
                hole_index = (hole_index + 1) % GameConstants.NUMBER_OF_HOLES
                if hole_index == starting_hole:
                    hole_index = (hole_index + 1) % GameConstants.NUMBER_OF_HOLES
                self.board[hole_index] += 1

            return hole_index
        
        def add_seeds_to_score(hole_index):
            self.player.score += self.board[hole_index]
            self.board[hole_index] = 0

        if move == "None":
            self.player.isstarved = True
            starting_hole_to_count_seeds = GameConstants.NUMBER_OF_HOLES_PER_PLAYER * self.opponent.camp
            self.player.score += sum(self.board[starting_hole_to_count_seeds:starting_hole_to_count_seeds + GameConstants.NUMBER_OF_HOLES_PER_PLAYER])
            return

        starting_hole = GameConstants.ALPHABET.index(move)

        last_hole_index = distribute_seeds(starting_hole)
        current_capture_hole = last_hole_index
        
        while (hole_in_enemy_row(current_capture_hole) and hole_has_required_seeds(current_capture_hole)):
            add_seeds_to_score(current_capture_hole)
            current_capture_hole = (current_capture_hole - 1) % GameConstants.NUMBER_OF_HOLES

        #now is_over() is invoked
    
    def is_over(self):
        """
        Determine if the game has ended.
        
        The game ends if:
        1. A player has lost (opponent has 25+ seeds).
        2. The total seeds on the board is too low (fewer than 7 seeds).
        3. The opponent is starved (cannot make a move, signaled by isstarved flag).
        
        Returns:
            bool: True if the game is over, False otherwise.
        """
        return ( self.lose() or
                  sum(self.board) < 7 or
                  self.opponent.isstarved )

    def show(self):
        """
        Display the current game state to the console using the external show_board function in interface.py.
        """
        show_board(self, GameConstants.NUMBER_OF_HOLES_PER_PLAYER, GameConstants.ALPHABET)

    def lose(self):
        """
        Check if the current player has lost the game.
        A player loses if their opponent has captured more than 24 seeds.
        Returns:
            bool: True if the current player has lost, False otherwise
        """
        return self.opponent.score > self.winning_score


if __name__ == "__main__":
    from easyAI import Human_Player, AI_Player, Negamax

    handle_arguments()

    scoring = lambda game: game.player.score - game.opponent.score
    
    # Create players based on configuration
    player1 = (Human_Player() if PlaygroundConstants.IS_HUMAN_PLAYER_1 
               else AI_Player(Negamax(PlaygroundConstants.AI_1_DEPTH, scoring)))
    player2 = (Human_Player() if PlaygroundConstants.IS_HUMAN_PLAYER_2 
               else AI_Player(Negamax(PlaygroundConstants.AI_2_DEPTH, scoring)))
    
    try:
        game = Mankala([player1, player2])
        game.play()
        
        score1 = game.players[0].score
        score2 = game.players[1].score

        if score1 == score2:
            show_draw()
        else:
            winner_idx, winning_score = (0, score1) if score1 > score2 else (1, score2)
            show_winner(winner_idx, winning_score)
            
    except ValueError as e:
        print(f"Game initialization failed: {e}")
        print("Please check your game configuration.")