def show_board(self):
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

