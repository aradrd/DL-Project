from utils.utils import Namespace
from torch.nn import MSELoss
from torch.optim import Adam, SGD
from models.vanilla import VanillaModel
from models.unet_rec import UNet_Rec
from ext.training import ReconstructionTrainer
from utils.utils import create_data_loaders
from datetime import datetime

class ModelRunner:
    str_to_model = {
        'vanilla': VanillaModel,
        'unet_rec': UNet_Rec,
    }

    str_to_optimizer = {
        'adam': Adam,
        'sgd': SGD,
    }

    str_to_loss = {
        'mse': MSELoss,
    }

    def __init__(self, args: dict):
        self.args = Namespace(**args)
        self.model = self.str_to_model[self.args.model](self.args.drop_rate, self.args.device, self.args.learn_mask).to(self.args.device)
        self.optimizer = self.str_to_optimizer[self.args.optimizer](self.model.parameters(), lr=self.args.lr)
        self.loss_fn = self.str_to_loss[self.args.loss]()
        self.trainer = ReconstructionTrainer(self.model, self.loss_fn, self.optimizer, self.args.device)
        self.train_loader, self.validation_loader, self.test_loader = create_data_loaders(self.args)

    def fit(self):
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{curr_time}]\tRunning with:\n{self.args}")
        fit_res = self.trainer.fit(self.train_loader, self.test_loader, num_epochs=self.args.num_epochs, print_every=self.args.report_interval, verbose=True)
        return fit_res