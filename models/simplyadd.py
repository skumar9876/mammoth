import copy
import math
import os
from sys import float_repr_style
import torch
from torch.nn import functional as F
from torch.optim import Adam, SGD

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer
from backbone.MNISTMLP import MNISTMLP


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' simply adding a prior.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--num_distill_steps', type=int, required=True)
    parser.add_argument('--buffer_minibatch_size', type=int, required=True)
    parser.add_argument('--distill_opt', type=str, required=True)
    parser.add_argument('--distill_lr', type=float, required=True)
    parser.add_argument('--prior_hidden_size', type=int, required=True)
    parser.add_argument('--net_hidden_size', type=int, required=True)
    parser.add_argument('--reinit_prior', action='store_true')
    return parser


class SimplyAdd(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SimplyAdd, self).__init__(backbone, loss, args, transform)
        self.net = MNISTMLP(28 * 28, 10, hidden_size=args.net_hidden_size)
        self.net_init = copy.deepcopy(self.net)
        self.prior = MNISTMLP(28 * 28, 10, hidden_size=args.prior_hidden_size)
        self.prior_old = copy.deepcopy(self.prior)
        if args.distill_opt == 'SGD':
            self.prior_opt = SGD(self.prior.parameters(), lr=self.args.distill_lr)
        else:
            self.prior_opt = Adam(self.prior.parameters(), lr=self.args.distill_lr)
        self.PRIOR_PATH = "prior_model.pt"
        self.num_distill_steps = args.num_distill_steps
        self.reinit_prior = args.reinit_prior
        self.step = 0

        # Add models to device.
        self.net.to(self.device)
        self.net_init.to(self.device)
        self.prior.to(self.device)
        self.prior_old.to(self.device)

        # Save initial parameters.
        self.TRAIN_INIT_PATH = "train_model_init.pt"

        self.buffer = Buffer(self.args.buffer_size, self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass using the full model.
        :param x: batch of inputs
        :return: the result of the computation
        """
        return self.prior_old(x) + self.net(x)

    def observe(self, inputs, labels, not_aug_inputs):
        if self.step == 0:
            torch.save(self.net_init.state_dict(), f'{self.model_save_dir}/{self.TRAIN_INIT_PATH}')
        self.step += 1
        # Update train network.
        self.opt.zero_grad()
        outputs = self.prior_old(inputs).detach() + self.net(inputs)
        train_loss = self.loss(outputs, labels)
        train_loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        # if self.step % self.update_period == 0:
        #     self.distill()
        #     self.update_prior()
        #     self.update_train()

        return train_loss.item()

    def distill(self):
        if not self.buffer.is_empty():
            for i in range(self.num_distill_steps):
                buf_inputs, _ = self.buffer.get_data(
                    self.args.buffer_minibatch_size, transform=self.transform)
                buf_pred_logits = self.prior(buf_inputs) + self.net_init(buf_inputs).detach()
                buf_target_logits = self.prior_old(buf_inputs).detach() + self.net(buf_inputs).detach()

                self.prior_opt.zero_grad()
                prior_loss = F.mse_loss(buf_pred_logits, buf_target_logits)

                try:
                    assert not math.isnan(prior_loss.item())
                except:
                    print("Prior loss is nan!")
                    import pdb; pdb.set_trace()

                prior_loss.backward()
                self.prior_opt.step()

    def update_prior(self):
        torch.save(self.prior.state_dict(), f'{self.model_save_dir}/{self.PRIOR_PATH}')
        self.prior_old.load_state_dict(torch.load(f'{self.model_save_dir}/{self.PRIOR_PATH}'), strict=True)
        if self.reinit_prior:
            self.prior.load_state_dict(torch.load(f'{self.model_save_dir}/{self.TRAIN_INIT_PATH}'), strict=True)

    def update_train(self):
        self.net.load_state_dict(torch.load(f'{self.model_save_dir}/{self.TRAIN_INIT_PATH}'), strict=True)
    
    def end_task(self, unused_dataset):
        self.distill()
        self.update_prior()
        self.update_train()
    
    def set_model_save_dir(self, model_save_dir):
        if not os.path.isdir(model_save_dir):
            os.makedir(model_save_dir)
        self.model_save_dir = model_save_dir