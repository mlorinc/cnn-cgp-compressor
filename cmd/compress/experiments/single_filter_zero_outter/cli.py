import argparse

def get_argument_parser(parser: argparse._SubParsersAction):
    parser.add_argument("--layer-names", nargs="+", default=["conv1", "conv2"], help="List of CNN layer names")
    parser.add_argument("--prefix", default="", help="Prefix for experiment names")
    parser.add_argument("--suffix", default="", help="Suffix for experiment names")
    parser.add_argument("--grid-sizes", nargs="+", type=int, default=[(5, 5)], help="List of grid sizes (rows, columns)")
    parser.add_argument("-n", default=30, type=int, help="Amount of filters to be tested")
    parser.add_argument("--reuse", type=str, default=None, help="Reuse experiment configuration from the other experiment")
    parser.add_argument("--name-format", type=str, default=None, help="Name format of the resued experiments")        
    return parser