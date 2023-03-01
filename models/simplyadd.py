import copy
import math
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
    parser.add_argument('--train_opt', type=str, required=True)
    parser.add_argument('--distill_opt', type=str, required=True)
    parser.add_argument('--distill_lr', type=float, required=True)
    parser.add_argument('--prior_hidden_size', type=int, required=True)
    parser.add_argument('--net_hidden_size', type=int, required=True)
    parser.add_argument('--reinit_prior', action='store_true')
    parser.add_argument('--selective_distill', action='store_true')
    parser.add_argument('--weight_decay_train', type=float, default=0.0)
    parser.add_argument('--weight_decay_prior', type=float, default=0.0)
    parser.add_argument('--distill_loss_type', type=str, default='MSE')
    parser.add_argument('--use_task_buffer', action='store_true')
    parser.add_argument('--regularize_train_net', type=float, default=0.0)
    parser.add_argument('--regularize_prior_net', type=float, default=0.0)
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
        self.prior_init = copy.deepcopy(self.prior)
        self.initialize_train_opt()
        self.initialize_prior_opt()
        
        self.PRIOR_PATH = "prior_model.pt"
        self.num_distill_steps = args.num_distill_steps
        self.reinit_prior = args.reinit_prior
        self.step = 0
        self.internal_task_id = 0

        # Add models to device.
        self.net.to(self.device)
        self.net_init.to(self.device)
        self.prior.to(self.device)
        self.prior_old.to(self.device)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        if self.args.use_task_buffer:
            self.task_buffer = Buffer(self.args.buffer_size, self.device)
        # TODO: Re-factor this. It's a variable used just for evaluation purposes.
        self.just_distilled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass using the full model.
        :param x: batch of inputs
        :return: the result of the computation
        """
        if self.just_distilled and self.args.selective_distill:
            return self.prior_old(x)
        else:
            return self.prior_old(x) + self.net(x)

    def observe(self, inputs, labels, not_aug_inputs):
        self.just_distilled = False

        self.step += 1
        # Update train network.
        self.opt.zero_grad()
        prior_outputs = self.prior_old(inputs).detach()
        outputs = prior_outputs + self.net(inputs)
        train_loss = self.loss(outputs, labels)

        internal_task_ids = self.internal_task_id * torch.ones(size=(len(inputs), 1))
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data, task_labels=internal_task_ids)
        
        if self.args.use_task_buffer:
            self.task_buffer.add_data(examples=not_aug_inputs, logits=outputs.data, task_labels=internal_task_ids)

        # Regularize the train network to output logits of zeros on both current task and previous task data.
        if self.args.regularize_train_net > 0 and self.internal_task_id > 0:
            reg_loss_task_data = F.mse_loss(outputs, prior_outputs)
            train_loss += self.args.regularize_train_net * reg_loss_task_data
            
            buf_inputs, buf_logits, buf_internal_task_ids = self.buffer.get_data(
                    self.args.buffer_minibatch_size, transform=self.transform)
            buf_pred_logits = self.prior_old(buf_inputs).detach() + self.net(buf_inputs)
            buf_target_logits = self.prior_old(buf_inputs).detach() + float(
                not self.args.selective_distill) * self.net_init(buf_inputs).detach()
            
            reg_mask = torch.ones_like(buf_internal_task_ids)  # (buf_internal_task_ids != self.internal_task_id).detach().float()
            reg_loss = reg_mask * F.mse_loss(buf_pred_logits, buf_target_logits, reduction='none')
            train_loss += self.args.regularize_train_net * reg_loss.mean(dim=-1).sum() / reg_mask.sum()
        
        train_loss.backward()
        self.opt.step()
        return train_loss.item()

    def distill(self):
        if not self.buffer.is_empty():
            for i in range(self.num_distill_steps):
                buf_inputs, buf_logits, buf_internal_task_ids = self.buffer.get_data(
                    self.args.buffer_minibatch_size, transform=self.transform)
                buf_pred_logits = self.prior(buf_inputs) + float(
                    not self.args.selective_distill) * self.net_init(buf_inputs).detach()
                buf_target_logits = self.prior_old(buf_inputs).detach() + (
                    buf_internal_task_ids == self.internal_task_id).detach().float() * self.net(buf_inputs).detach()

                self.prior_opt.zero_grad()
                if self.args.distill_loss_type == 'MSE':
                    prior_loss = F.mse_loss(buf_pred_logits, buf_target_logits)
                elif self.args.distill_loss_type == 'KL':
                    loss_fn = torch.nn.KLDivLoss(log_target=True)
                    prior_loss = loss_fn(
                        F.log_softmax(buf_pred_logits, dim=-1), 
                        F.log_softmax(buf_target_logits, dim=-1))
                try:
                    assert not math.isnan(prior_loss.item())
                except:
                    print("Prior loss is nan!")
                    import pdb; pdb.set_trace()
                
                if self.args.use_task_buffer:
                    buf_inputs, buf_logits, buf_internal_task_ids = self.task_buffer.get_data(
                        self.args.buffer_minibatch_size, transform=self.transform)
                    buf_pred_logits = self.prior(buf_inputs) + float(
                        not self.args.selective_distill) * self.net_init(buf_inputs).detach()
                    buf_target_logits = self.prior_old(buf_inputs).detach() + self.net(buf_inputs).detach()

                    if self.args.distill_loss_type == 'MSE':
                        reg_loss = F.mse_loss(buf_pred_logits, buf_target_logits)
                    elif self.args.distill_loss_type == 'KL':
                        loss_fn = torch.nn.KLDivLoss(log_target=True)
                        reg_loss = loss_fn(
                            F.log_softmax(buf_pred_logits, dim=-1), 
                            F.log_softmax(buf_target_logits, dim=-1))
                    try:
                        assert not math.isnan(prior_loss.item())
                    except:
                        print("Prior loss is nan!")
                        import pdb; pdb.set_trace()

                    prior_loss += self.args.regularize_prior_net * reg_loss

                prior_loss.backward()
                self.prior_opt.step()
                    

    def update_prior(self):
        self.prior_old.load_state_dict(self.prior.state_dict(), strict=True)
        if self.reinit_prior:
            self.prior.load_state_dict(self.prior_init.state_dict(), strict=True)
            self.initialize_prior_opt()

    def update_train(self):
        self.net.load_state_dict(self.net_init.state_dict(), strict=True)
        self.initialize_train_opt()

    def initialize_train_opt(self):
        if self.args.train_opt == 'SGD':
            self.opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay_train)
        elif self.args.train_opt == 'Adam':
            self.opt = Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay_train)
    
    def initialize_prior_opt(self):
        if self.args.distill_opt == 'SGD':
            self.prior_opt = SGD(self.prior.parameters(), lr=self.args.distill_lr, weight_decay=self.args.weight_decay_prior)
        elif self.args.distill_opt == 'Adam':
            self.prior_opt = Adam(self.prior.parameters(), lr=self.args.distill_lr, weight_decay=self.args.weight_decay_prior)
    
    def end_task(self, unused_dataset):
        self.distill()
        self.update_prior()
        self.update_train()

        if self.args.selective_distill:
            self.internal_task_id += 1
            self.just_distilled = True

        if self.args.use_task_buffer:
            self.task_buffer.empty()