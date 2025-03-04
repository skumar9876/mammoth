# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
from argparse import Namespace
from typing import Tuple

import scipy
import sklearn.metrics as metrics
import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.status import ProgressBar

try:
    import wandb
except ImportError:
    wandb = None

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    if hasattr(model, 'prior_net'):
        model.prior_net.eval()
    
    results_dict = {}

    accs, accs_mask_classes = [], []

    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        logits_arr, labels_arr = [], []
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                logits_arr.append(outputs.detach().data.cpu())
                labels_arr.append(labels.detach().cpu())

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        logits_arr = torch.cat(logits_arr)
        probs_arr = torch.softmax(logits_arr, axis=-1).numpy()
        logits_arr = logits_arr.numpy()
        labels_arr = torch.cat(labels_arr).numpy()
        results_dict[f'auroc/{str(k).zfill(2)}'] = metrics.roc_auc_score(labels_arr, probs_arr, multi_class='ovr')
        results_dict[f'acc/{str(k).zfill(2)}'] = metrics.accuracy_score(labels_arr, np.argmax(logits_arr, 1))

        results_dict[f'logits/mean_{str(k).zfill(2)}'] = np.mean(logits_arr)
        results_dict[f'logits/var_{str(k).zfill(2)}'] = np.var(logits_arr)
        results_dict[f'entropy/{str(k).zfill(2)}'] = np.mean(scipy.stats.entropy(probs_arr, axis=-1, base=10))

        for name, param in model.named_parameters():
            if param.requires_grad:
                results_dict[f'l2_norm/{name}'] = torch.sqrt(torch.sum(param ** 2)).detach().cpu()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes, results_dict


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task, _ = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        if hasattr(model, 'prior_net'):
            model.prior_net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics:
            accs, accs_mask_classes, _ = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs_mask_classes

        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.args.n_epochs):
            if args.model == 'joint':
                continue
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs)
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if scheduler is not None:
                scheduler.step()

        # Evaluate before distilling.
        accs_b4_distill, accs_mask_classes_b4_distill, results_dict_b4_distill = evaluate(model, dataset)
        results_dict_b4_distill = {f'b4_distill_{key}': val for key, val in results_dict_b4_distill.items()}
        mean_acc_b4_distill = np.mean(accs_b4_distill)
        mean_acc_mask_classes_b4_distill = np.mean(accs_mask_classes_b4_distill)
        if not args.disable_log:
            logger.log([mean_acc_b4_distill, mean_acc_mask_classes_b4_distill], before_distill=True)
            logger.log_fullacc([accs_b4_distill, accs_mask_classes_b4_distill], before_distill=True)


        if hasattr(model, 'end_task'):
            # Distill.
            model.end_task(dataset)

        accs, accs_mask_classes, results_dict = evaluate(model, dataset)
        results.append(accs)
        results_mask_classes.append(accs_mask_classes)

        mean_acc = np.mean(accs)
        mean_acc_mask_classes = np.mean(accs_mask_classes)
        print_mean_accuracy([mean_acc, mean_acc_mask_classes], t + 1, dataset.SETTING)

        if not args.disable_log:
            logger.log([mean_acc, mean_acc_mask_classes])
            logger.log_fullacc([accs, accs_mask_classes])

        if not args.nowand:
            d2={'RESULT_class_mean_accs': mean_acc, 'RESULT_task_mean_accs': mean_acc_mask_classes,
                **{f'RESULT_class_acc/{str(i).zfill(2)}': a for i, a in enumerate(accs)},
                **{f'RESULT_task_acc/{str(i).zfill(2)}': a for i, a in enumerate(accs_mask_classes)}}

            d2_b4_distill={'RESULT_class_mean_accs_b4_distill': mean_acc_b4_distill, 'RESULT_task_mean_accs_b4_distill': mean_acc_mask_classes_b4_distill,
                           **{f'RESULT_class_acc_b4_distill/{str(i).zfill(2)}': a for i, a in enumerate(accs_b4_distill)}}

            d2.update(d2_b4_distill)
            d2.update(results_dict_b4_distill)
            d2.update(results_dict)

            wandb.log(d2)



    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                    results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
