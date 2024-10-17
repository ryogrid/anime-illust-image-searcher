import argparse
import sys

import tagging
import genmodel
import counttag

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('command', nargs=1, help='command to run')
    # dummy
    parser.add_argument('--dir', nargs=1, help='')
    # dummy
    parser.add_argument('--dim', nargs=1, type=int, help='')
    args: argparse.Namespace = parser.parse_args()

    if args.command[0] == 'tagging':
        tagging.main(sys.argv[2:])
    elif args.command[0] == 'genmodel':
        genmodel.main(sys.argv[2:])
    elif args.command[0] == 'counttag':
        counttag.main()
    else:
        print('Invalid command')
        exit(1)

main()