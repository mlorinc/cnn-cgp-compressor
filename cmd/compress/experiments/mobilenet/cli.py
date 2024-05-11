import argparse

thresholds = [250, 150, 100, 50, 25, 15, 10, 0]

def get_argument_parser(parser: argparse._SubParsersAction):
    parser.add_argument("--input-layer-name", default="conv1", help="List of the input CNN layer names")
    parser.add_argument("--output-layer-name", default="conv2", help="List of the output CNN layer names")
    parser.add_argument("-a", "--generate-all", action="store_true", help="List of the output CNN layer names")
    parser.add_argument("--prefix", default="", help="Prefix for experiment names")
    parser.add_argument("--suffix", default="", help="Suffix for experiment names")
    parser.add_argument("--mse-thresholds", nargs="+", type=float, default=thresholds, help="List of MSE thresholds")
    parser.add_argument("--rows", type=int, default=30, help="Number of rows per filter")
    parser.add_argument("--cols", type=int, default=7, help="Number of columns per layer")
    return parser
