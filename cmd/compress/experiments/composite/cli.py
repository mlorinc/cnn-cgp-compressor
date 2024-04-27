import argparse

def get_argument_parser(parser: argparse._SubParsersAction):
    parser.add_argument("-b", "--batches", default=None, type=int, help="Split single experiment into multiple smaller batches")    
    return parser