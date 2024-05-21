import random
import torch
import numpy as np
from json import load
from ext.plot import plot_fit
from utils.config import create_arg_parser
from utils.runner import ModelRunner
from pathlib import Path

def main():
    # Get arguments from cmd / defaults
    args = create_arg_parser().parse_args()
    if args.config:
        for path in args.config:
            with open(path, 'r') as f:
                config = load(f)
                args.__dict__.update(config)
    args.data_path = f"{Path(args.base_path) / args.data_path}"

    # Freeze seeds for result reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    runner = ModelRunner(vars(args))
    fit_res = runner.fit()
    
    plot_fit(fit_res)

if __name__ == "__main__":
    main()
