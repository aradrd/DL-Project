import argparse
import shutil
import random
import torch
import numpy as np
from matplotlib import pyplot as plt
from models.vanilla import VanillaModel
from models.unet_rec import UNet_Rec
from utils.utils import create_data_loaders, freq_to_image
from ext.training import ReconstructionTrainer
from ext.plot import plot_fit

str_to_model = {
    'vanilla': VanillaModel,
    'unet_rec': UNet_Rec
}

str_to_optimizer = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

str_to_loss = {
    'mse': torch.nn.MSELoss,
}



def main():
    args = create_arg_parser().parse_args() #get arguments from cmd/defaults
    train_loader, validation_loader, test_loader = create_data_loaders(args) #get dataloaders


    #freeze seeds for result reproducability
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = str_to_model[args.model](args.drop_rate, args.device, args.learn_mask).to(args.device) #Example instatiation - replace with your model
    optimizer = str_to_optimizer[args.optimizer](model.parameters(), lr=args.lr) #initialize optimizer
    loss_fn = str_to_loss[args.loss]()
    trainer = ReconstructionTrainer(model, loss_fn, optimizer, args.device)
    fit_res = trainer.fit(train_loader, test_loader, num_epochs=args.num_epochs, print_every=args.report_interval, verbose=True)
    plot_fit(fit_res)
    print("ima shcha zona")







def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='Random Seed for reproducability.')
    parser.add_argument('--data-path', type=str, default='/datasets/fastmri_knee/', help='path to MRI dataset.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Pass "cuda" to use gpu')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of threads used for data handling.')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--report-interval', type=int, default=10, help='Report training stats once per this much iterations.')
    parser.add_argument('--drop-rate', type=float, default=0.8, help='Percentage of data to drop from each image (dropped in freq domain).')
    parser.add_argument('--learn-mask', action='store_true', default=False, help='Whether to learn subsampling mask')
    parser.add_argument('--results-root', type=str, default='results', help='result output dir.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learn rate for your reconstruction model.')
    parser.add_argument('--mask-lr', type=float, default=0.01, help='Learn rate for your mask (ignored if the learn-mask flag is off).')
    parser.add_argument('--val-test-split', type=float, default=0.3, help='Portion of test set (NOT of the entire dataset, since train-test split is pre-defined) to be used for validation.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use for training.')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function to use for training.')
    parser.add_argument('--model', type=str, default='unet_rec', help='Model to use for training.')

    return parser

if __name__ == "__main__":
    main()
