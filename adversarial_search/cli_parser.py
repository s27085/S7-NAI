import argparse
from constants import GameConstants, PlaygroundConstants

def handle_arguments():
    """
    Parses command-line arguments provided by the user and updates the
    global GameConstants and PlaygroundConstants classes accordingly.

    This function configures the game's rules (via GameConstants) and 
    the player settings (via PlaygroundConstants, determining Human/AI
    status and AI search depth).

    usage: game.py [options] p1 p2

    positional arguments:
    p1 {ai,human}         type of an agent
    p2 {ai,human}         type of an agent

    options:
    -h, --help            show this help message and exit
    -n SEEDS, --seeds SEEDS
                            number of seeds in holes at the beggining
    -ss STARTING_SCORE, --starting-score STARTING_SCORE
                            score that each player starts with
    -ph PLAYER_HOLES, --player-holes PLAYER_HOLES
                            number of holes each player has
    -ws WINNING_SCORE, --winning-score WINNING_SCORE
                            a score that players aim to achieve
    -a1 A1                difficulty of ai1, if applicable
    -a2 A2                difficulty of ai2, if applicable
    
    Side Effects:
        Modifies attributes of GameConstants and PlaygroundConstants in place.
    """
        
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

    GameConstants.STARTING_SEEDS_IN_HOLE = args.seeds
    GameConstants.NUMBER_OF_HOLES_PER_PLAYER = args.player_holes

    GameConstants.NUMBER_OF_HOLES = GameConstants.NUMBER_OF_HOLES_PER_PLAYER * 2

    GameConstants.STARTING_SCORE = args.starting_score
    GameConstants.WINNING_SCORE = args.winning_score