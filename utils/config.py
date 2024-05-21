from json import load
from pathlib import Path
from argparse import ArgumentParser
from torch.cuda import is_available as cuda_available
from typing import List

DEFAULT_CONFIG = Path(__file__).parent / '../configs/default.json'

str_to_type = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}

def create_arg_parser(config=DEFAULT_CONFIG):
    parser = ArgumentParser()
    args_dict = {}
    with open(config, 'r') as f:
        args_dict = load(f)

    for arg, value in args_dict.items():
        def_val = value["default"]
        help_str = value["help"]
        action = value.get("action", "store")
        arg_type = value.get("type", None)
        arg_type = str_to_type[arg_type] if arg_type is not None else None
        parser.add_argument(f'--{arg}', default=def_val, help=help_str, action=action, type=arg_type)

    parser.set_defaults(device="cuda" if cuda_available() else "cpu")

    return parser
