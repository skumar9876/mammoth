import copy
from sys import float_repr_style
import torch
from torch.nn import functional as F
from torch.optim import SGD

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' simply adding a prior.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--update_period', type=int, required=True)
    return parser


class SimplyAdd(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        self.prior = copy.deepcopy(backbone)
        self.prior_old = copy.deepcopy(self.prior)
        self.prior_opt = SGD(self.prior.parameters(), lr=self.args.lr)
        self.PRIOR_PATH = "prior_model.pt"
        self.TRAIN_PATH = "train_model.pt"
        self.update_period = args.update_period
        self.step = 0

        super(SimplyAdd, self).__init__(backbone, loss, args, transform)
        self.net_old = copy.deepcopy(self.net)

        # Add models to device.
        self.net.to(self.device)
        self.net_old.to(self.device)
        self.prior.to(self.device)
        self.prior_old.to(self.device)

        # Save initial parameters.
        self.TRAIN_INIT_PATH = "train_model_init"
        torch.save(self.net(), self.TRAIN_INIT_PATH)

        self.buffer = Buffer(self.args.buffer_size, self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass using the full model.
        :param x: batch of inputs
        :return: the result of the computation
        """
        return self.prior(x) + self.net(x)

    def observe(self, inputs, labels, not_aug_inputs):
        self.step += 1

        # Update prior network.
        if not self.buffer.is_empty():
            buf_inputs, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_pred_logits = self.prior(buf_inputs)
            buf_target_logits = self.prior_old(buf_inputs).detach() + self.net_old(buf_inputs).detach()

            self.prior_opt.zero_grad()
            prior_loss = F.mse_loss(buf_pred_logits, buf_target_logits)
            prior_loss.backward()
            self.prior_opt.step()
    
        # Update train network.
        self.opt.zero_grad()
        outputs = self.prior_old(inputs).detach() + self.net(inputs)
        train_loss = self.loss(outputs, labels)
        train_loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        if self.step % self.update_period == 0:
            self.update_prior()
            self.update_train()

        return train_loss.item()
    
    def update_prior(self):
        torch.save(self.prior.state_dict(), self.PRIOR_PATH)
        self.prior_old.load_state_dict(torch.load(self.PRIOR_PATH), strict=False)

    def update_train(self):
        torch.save(self.net(), self.TRAIN_PATH)
        self.net_old.load_state_dict(torch.load(self.TRAIN_PATH), strict=False)
        self.net.load_state_dict(torch.load(self.TRAIN_INIT_PATH), strict=False)