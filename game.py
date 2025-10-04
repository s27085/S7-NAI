from easyAI import TwoPlayerGame

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
        for i, player in enumerate(players):
            player.score = 0
            player.isstarved = False
            player.camp = i
        self.players = players
        
        self.board = [4, 4, 4, 4, 4, 4,  
                      4, 4, 4, 4, 4, 4]  
                      
        self.current_player = 1

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
        if move == "None":
            self.player.isstarved = True
            s = 6 * self.opponent.camp
            self.player.score += sum(self.board[s:s + 6])
            return

        move = 'abcdefghijkl'.index(move)

        pos = move
        for i in range(self.board[move]):
            pos = (pos + 1) % 12
            if pos == move:
                pos = (pos + 1) % 12
            self.board[pos] += 1

        self.board[move] = 0

        while ((pos // 6) == self.opponent.camp
               and (self.board[pos] in [2, 3])):
            self.player.score += self.board[pos]
            self.board[pos] = 0
            pos = (pos - 1) % 12

    def possible_moves(self):
        """
        Check for possible moves for the current player.
        
        Returns a list of valid moves (holes) that the current player can play.
        If no valid moves are available, returns ['None'].
        A player must play any hole that contains enough seeds to
        'feed' the opponent. If no hole has this many seeds, any
        non-empty hole can be played.
        Returns:
            list: List of valid moves as strings ('a'-'l') or ['None']
            if no moves are available

        Side Effects:
            - None
        Note: The current player is determined by self.current_player
        which is either 1 or 2, representing the two players.
        """
        if self.current_player == 1:
            if max(self.board[:6]) == 0: return ['None']
            moves = [i for i in range(6) if (self.board[i] >= 6 - i)]
            if moves == []:
                moves = [i for i in range(6) if self.board[i] != 0]
        else:
            if max(self.board[6:]) == 0: return ['None']
            moves = [i for i in range(6,12) if (self.board[i] >= 12-i)]
            if moves == []:
                moves = [i for i in range(6, 12) if self.board[i] != 0]

        return ['abcdefghijkl'[u] for u in moves]

    def show(self):
        """
        Display the current game state to the console.
        Prints the board layout and player scores in a human-readable format.
        Side Effects:
            - Outputs to the console
        """
        print("Score: %d / %d" % tuple(p.score for p in self.players))
        print('  '.join('lkjihg'))
        print(' '.join(["%02d" % i for i in self.board[-1:-7:-1]]))
        print(' '.join(["%02d" % i for i in self.board[:6]]))
        print('  '.join('abcdef'))

    def lose(self):
        """
        Check if the current player has lost the game.
        A player loses if their opponent has captured more than 24 seeds.
        Returns:
            bool: True if the current player has lost, False otherwise
        """
        return self.opponent.score > 24

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
    game = Mankala([AI_Player(ai), AI_Player(ai_2)])

    game.play()

    if game.players[0].score > game.players[1].score:
        print("Player 1 wins with a score of %d." % game.players[0].score)
    elif game.players[0].score < game.players[1].score:
        print("Player 2 wins with a score of %d." % game.players[1].score)
    else:
        print("Draw.")