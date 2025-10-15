import argparse
from constants import GameConstants, PlaygroundConstants

def handle_arguments():
    parser = argparse.ArgumentParser(
        prog = 'Mankala (easyAI implementation)',
        epilog = 'Enjoy your game!',
        usage = 'game.py [options] p1 p2' 
    )
    parser.add_argument('-n', '--seeds', default=GameConstants.STARTING_SEEDS_IN_HOLE, type=int,
                        help="number of seeds in holes at the beggining")
    parser.add_argument('-ss', '--starting-score', default=GameConstants.STARTING_SCORE, type=int,
                        help="score that each player starts with")
    parser.add_argument('-ph', '--player-holes', default=GameConstants.NUMBER_OF_HOLES_PER_PLAYER, type=int,
                        help="number of holes each player has")
    parser.add_argument('-ws', '--winning-score', default=GameConstants.WINNING_SCORE , type=int,
                        help="a score that players aim to achieve")
    
    parser.add_argument('p1', metavar='p1 {ai,human}', choices=['ai', 'human'],
                        help="type of an agent")
    parser.add_argument('p2', metavar='p2 {ai,human}', choices=['ai', 'human'],
                        help="type of an agent")
    
    parser.add_argument('-a1', default=PlaygroundConstants.AI_1_DEPTH, type=int,
                        help="difficulty of ai1, if applicable")
    parser.add_argument('-a2', default=PlaygroundConstants.AI_2_DEPTH, type=int,
                        help="difficulty of ai2, if applicable")
    
    args = parser.parse_args()

    PlaygroundConstants.AI_1_DEPTH = args.a1
    PlaygroundConstants.AI_2_DEPTH = args.a2
    PlaygroundConstants.IS_HUMAN_PLAYER_1 = True if args.p1 == 'human' else False
    PlaygroundConstants.IS_HUMAN_PLAYER_2 = True if args.p2 == 'human' else False
    print(f'Player 1 is human in parser: {PlaygroundConstants.IS_HUMAN_PLAYER_1}')
    print(f'Player 2 is human in parser: {PlaygroundConstants.IS_HUMAN_PLAYER_2}')

    GameConstants.STARTING_SEEDS_IN_HOLE = args.seeds
    GameConstants.NUMBER_OF_HOLES_PER_PLAYER = args.player_holes

    GameConstants.NUMBER_OF_HOLES = GameConstants.NUMBER_OF_HOLES_PER_PLAYER * 2

    GameConstants.STARTING_SCORE = args.starting_score
    GameConstants.WINNING_SCORE = args.winning_score