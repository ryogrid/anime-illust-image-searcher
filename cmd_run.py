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

    if args.command[0] == 'tagging':
        tagging.main(sys.argv[2:])
    elif args.command[0] == 'genmodel':
        genmodel.main(sys.argv[2:])
    else:
        print('Invalid command')
        exit(1)

main()
