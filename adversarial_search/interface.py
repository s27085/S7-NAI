def show_board(self):
    """
    Display the current game state to the console.
    Prints the board layout and player scores in a human-readable format.
    Side Effects:
        - Outputs to the console
    """
    row_output = '  '.join('LKJIHG')
    print(row_output + "     Score")

    row_output = ' '.join(['{:02d}'.format(i) for i in self.board[-1:-7:-1]])
    print(row_output + f"  | Player 2: \033[31m" + str(self.players[1].score) + '\033[0m')

    row_output = ' '.join(['{:02d}'.format(i) for i in self.board[:6]])
    print(row_output + f"  | Player 1: \033[34m" + str(self.players[0].score) + '\033[0m')


    print('  '.join('ABCDEF'))


