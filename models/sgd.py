# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.utils.continual_model import ContinualModel
from torch.optim import Adam, SGD
from utils.args import add_management_args, add_experiment_args, ArgumentParser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--train_opt', type=str, default='SGD')
    parser.add_argument('--weight_decay_train', type=float, default=0.0)
    return parser


class Sgd(ContinualModel):
    NAME = 'sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Sgd, self).__init__(backbone, loss, args, transform)

        if self.args.train_opt == 'Adam':
            self.opt = Adam(self.net.parameters(), lr=self.args.lr, weight_decay=args.weight_decay_train)
        elif self.args.train_opt == 'SGD':
            self.opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay_train)
        

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
