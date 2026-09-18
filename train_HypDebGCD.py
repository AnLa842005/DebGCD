"""Mechanical Hyp-DebGCD proof-of-concept training entrypoint.

DebGCD training logic is combined with the hyperbolic representation-learning
and optimizer strategy from Visual-AI/HypCD's train_HypSimGCD.py.
"""

import argparse
import os
import sys
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, lr_scheduler

import geoopt.optim.radam as radam_

from hyptorch.pmath import dist_matrix
from model import DistillLoss, ContrastiveLearningViewGenerator, HypDebGCDHead


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



class SupConLoss(torch.nn.Module):
    """Supervised contrastive loss with cosine or hyperbolic-distance similarity."""

    def __init__(
        self,
        temperature=0.07,
        contrast_mode='all',
        base_temperature=0.07,
        hyp_c=0,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.hyp_c = hyp_c

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError(
                '`features` needs to be [bsz, n_views, ...], '
                'at least 3 dimensions are required'
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        if self.hyp_c == 0:
            anchor_dot_contrast = torch.div(
                torch.matmul(
                    F.normalize(anchor_feature, dim=-1, p=2),
                    F.normalize(contrast_feature, dim=-1, p=2).T,
                ),
                self.temperature,
            )
        else:
            anchor_dot_contrast = torch.div(
                -dist_matrix(anchor_feature, contrast_feature, c=self.hyp_c),
                self.temperature,
            )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(
                batch_size * anchor_count,
                device=device,
            ).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


def info_nce_logits(
    features,
    n_views=2,
    temperature=1.0,
    hyp_c=0,
    normalize=True,
):
    device = features.device
    batch_size = int(features.size(0) / n_views)

    labels = torch.cat(
        [torch.arange(batch_size, device=device) for _ in range(n_views)],
        dim=0,
    )
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    if normalize:
        features = F.normalize(features, dim=1)

    if hyp_c == 0:
        similarity_matrix = torch.matmul(
            F.normalize(features, dim=-1, p=2),
            F.normalize(features, dim=-1, p=2).T,
        )
    else:
        similarity_matrix = -dist_matrix(features, features, c=hyp_c)

    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(
        similarity_matrix.shape[0],
        -1,
    )

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(
        similarity_matrix.shape[0],
        -1,
    )

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
    logits = logits / temperature
    return logits, labels


def split_optimizer_parameters(student, head):
    hyperbolic_parameter_groups = [
        [
            parameter
            for parameter in head.hyperbolic_projector.parameters()
            if parameter.requires_grad
        ],
        [
            parameter
            for parameter in head.last_layer.parameters()
            if parameter.requires_grad
        ],
        [
            parameter
            for parameter in head.last_layer_deb.parameters()
            if parameter.requires_grad
        ],
    ]
    hyperbolic_parameters = [
        parameter
        for group in hyperbolic_parameter_groups
        for parameter in group
    ]
    hyperbolic_parameter_ids = {
        id(parameter)
        for parameter in hyperbolic_parameters
    }
    assert len(hyperbolic_parameter_ids) == len(hyperbolic_parameters), (
        'Hyperbolic parameter groups contain shared parameters'
    )

    regularized = []
    not_regularized = []
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad or id(parameter) in hyperbolic_parameter_ids:
            continue
        if name.endswith('.bias') or len(parameter.shape) == 1:
            not_regularized.append(parameter)
        else:
            regularized.append(parameter)

    sgd_parameters = regularized + not_regularized
    sgd_parameter_ids = {id(parameter) for parameter in sgd_parameters}
    trainable_parameter_ids = {
        id(parameter)
        for parameter in student.parameters()
        if parameter.requires_grad
    }

    assert sgd_parameter_ids.isdisjoint(hyperbolic_parameter_ids), (
        'SGD and RiemannianAdam parameter sets overlap'
    )
    assert sgd_parameter_ids | hyperbolic_parameter_ids == trainable_parameter_ids, (
        'Some trainable parameters are missing from the optimizers'
    )

    sgd_parameter_groups = [
        {'params': regularized},
        {'params': not_regularized, 'weight_decay': 0.},
    ]
    return sgd_parameter_groups, hyperbolic_parameter_groups


def optimizer_parameter_ids(optimizer):
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group['params']
    }


def assert_optimizer_parameter_sets_disjoint(optimizer, optimizer_hyper):
    optimizer_ids = optimizer_parameter_ids(optimizer)
    optimizer_hyper_ids = optimizer_parameter_ids(optimizer_hyper)
    assert optimizer_ids.isdisjoint(optimizer_hyper_ids), (
        'SGD and RiemannianAdam parameter sets overlap'
    )


def build_optimizers(student, head, args):
    params_groups, hyperbolic_parameter_groups = split_optimizer_parameters(
        student,
        head,
    )
    optimizer = SGD(
        params_groups,
        lr=args.lr * (args.batch_size / 128),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    optimizer_hyper = radam_.RiemannianAdam(
        [
            {'params': parameter_group}
            for parameter_group in hyperbolic_parameter_groups
        ],
        lr=0.01,
        stabilize=10,
    )
    assert_optimizer_parameter_sets_disjoint(optimizer, optimizer_hyper)
    return optimizer, optimizer_hyper


def ova_loss(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0), logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0), device=label.device).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :] + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
    Lo = open_loss_neg + open_loss
    return Lo


def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open * torch.log(logits_open + 1e-8), 1), 1))
    return Le


def compute_hyp_debgcd_loss(
    student_proj,
    student_out,
    student_ood,
    pseudo_out,
    class_labels,
    mask_lab,
    epoch,
    args,
    cluster_criterion,
):
    teacher_out = student_out.detach()
    loss = student_out.new_zeros(())
    components = {}

    # ---------------------------------- SimGCD ----------------------------------
    # supervised GCD loss
    sup_logits = torch.cat(
        [features[mask_lab] for features in (student_out / 0.1).chunk(2)],
        dim=0,
    )
    sup_labels = torch.cat(
        [class_labels[mask_lab] for _ in range(2)],
        dim=0,
    )
    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
    loss += args.sup_weight * cls_loss
    components['cls_loss'] = cls_loss

    # unsupervised GCD loss
    cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
    avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
    me_max_loss = (
        -torch.sum(torch.log(avg_probs ** (-avg_probs)))
        + math.log(float(len(avg_probs)))
    )
    cluster_loss = cluster_loss + args.memax_weight * me_max_loss
    loss += (1 - args.sup_weight) * cluster_loss
    components['me_max_loss'] = me_max_loss
    components['cluster_loss'] = cluster_loss

    # hyperbolic representation learning, unsupervised distance
    contrastive_logits, contrastive_labels = info_nce_logits(
        features=student_proj,
        hyp_c=args.c,
        normalize=False,
    )
    contrastive_loss_distance = nn.CrossEntropyLoss()(
        contrastive_logits,
        contrastive_labels,
    )

    # hyperbolic representation learning, unsupervised angle
    contrastive_logits_angle, contrastive_labels_angle = info_nce_logits(
        features=student_proj,
        hyp_c=0,
        normalize=False,
        temperature=args.hyper_temp_scale * 1.0,
    )
    contrastive_loss_angle = nn.CrossEntropyLoss()(
        contrastive_logits_angle,
        contrastive_labels_angle,
    )

    # hyperbolic representation learning, supervised distance and angle
    student_proj_sup = torch.cat(
        [
            features[mask_lab].unsqueeze(1)
            for features in student_proj.chunk(2)
        ],
        dim=1,
    )
    sup_con_labels = class_labels[mask_lab]
    sup_con_loss_distance = SupConLoss(hyp_c=args.c)(
        student_proj_sup,
        labels=sup_con_labels,
    )
    sup_con_loss_angle = SupConLoss(
        hyp_c=0,
        temperature=0.07 * args.hyper_temp_scale,
    )(
        student_proj_sup,
        labels=sup_con_labels,
    )

    loss_distance = (
        (1 - args.sup_weight) * contrastive_loss_distance
        + args.sup_weight * sup_con_loss_distance
    )
    loss_angle = (
        (1 - args.sup_weight) * contrastive_loss_angle
        + args.sup_weight * sup_con_loss_angle
    )

    lambda_distance = (
        epoch - (args.hyper_start_epoch - 1)
    ) / (
        (args.hyper_end_epoch - 1)
        - (args.hyper_start_epoch - 1)
    )
    lambda_distance = torch.max(torch.tensor([0, lambda_distance])).item()
    lambda_distance = torch.min(torch.tensor([1, lambda_distance])).item()
    lambda_distance *= args.hyper_max_weight

    loss_rep = (
        (1 - lambda_distance) * loss_angle
        + lambda_distance * loss_distance
    )
    loss += loss_rep

    components['contrastive_loss_distance'] = contrastive_loss_distance
    components['contrastive_loss_angle'] = contrastive_loss_angle
    components['sup_con_loss_distance'] = sup_con_loss_distance
    components['sup_con_loss_angle'] = sup_con_loss_angle
    components['loss_distance'] = loss_distance
    components['loss_angle'] = loss_angle
    components['lambda_distance'] = student_out.new_tensor(lambda_distance)
    components['loss_rep'] = loss_rep

    # ---------------------------------- Semantic Distribution Learning ----------------------------------
    # reference: https://github.com/VisionLearningGroup/OP_Match
    student_ood_scaled = student_ood / 0.1
    logits_ood = torch.cat(
        [features[mask_lab] for features in student_ood_scaled.chunk(2)],
        dim=0,
    )
    logits_ood_u = torch.cat(
        [features[~mask_lab] for features in student_ood_scaled.chunk(2)],
        dim=0,
    )
    logits_open_u1, logits_open_u2 = logits_ood_u.chunk(2)

    # SDL Loss for labeled samples
    sup_sdl_loss = ova_loss(logits_ood, sup_labels)

    # SDL Loss for unlabeled samples
    # entropy minimization
    L_oem = ova_ent(logits_open_u1) / 2.
    L_oem += ova_ent(logits_open_u2) / 2.

    # Soft consistency regularization
    logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
    logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
    logits_open_u1 = F.softmax(logits_open_u1, 1)
    logits_open_u2 = F.softmax(logits_open_u2, 1)
    unsup_sdl_loss = (
        0.1 * L_oem
        + torch.mean(
            torch.sum(
                torch.sum(torch.abs(logits_open_u1 - logits_open_u2) ** 2, 1),
                1,
            )
        )
    )
    loss += args.sdl_loss_weight * (sup_sdl_loss + unsup_sdl_loss)
    components['ova_entropy_loss'] = L_oem
    components['sup_sdl_loss'] = sup_sdl_loss
    components['unsup_sdl_loss'] = unsup_sdl_loss

    # distribution certainty score from OVA classifier
    ova_scores = F.softmax(
        student_ood_scaled.view(student_ood_scaled.size(0), 2, -1),
        1,
    ).detach()
    pred_close = ova_scores[:, 1, :].data.max(1)[1]
    tmp_range = torch.arange(
        0,
        ova_scores.size(0),
        dtype=torch.long,
        device=ova_scores.device,
    )
    unk_score = ova_scores[tmp_range, 0, pred_close]
    unk_score_unlabelled = torch.cat(
        [score[~mask_lab] for score in unk_score.chunk(2)],
        dim=0,
    )
    ood_cer_score = torch.abs(2 * unk_score_unlabelled - 1)

    # ---------------------------------- Auxiliary Debiased Learning ----------------------------------
    sup_logits_pseudo = torch.cat(
        [
            features[mask_lab]
            for features in (pseudo_out / args.pseudo_temp).chunk(2)
        ],
        dim=0,
    )
    sup_adl_loss = nn.CrossEntropyLoss()(sup_logits_pseudo, sup_labels)
    loss += (
        args.adl_loss_weight
        * (1 - args.pl_loss_weight)
        * sup_adl_loss
    )

    unsup_logits_pseudo = torch.cat(
        [
            features[~mask_lab]
            for features in (pseudo_out / args.pseudo_temp).chunk(2)
        ],
        dim=0,
    )
    unsup_logits = torch.cat(
        [
            features[~mask_lab]
            for features in (student_out / 0.1).chunk(2)
        ],
        dim=0,
    )
    pseudo_label = torch.softmax(unsup_logits.detach(), dim=-1)
    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(args.threshold).float()

    unsup_adl_loss = (
        F.cross_entropy(
            unsup_logits_pseudo,
            targets_u,
            reduction='none',
        )
        * mask
        * ood_cer_score
    ).mean()
    loss += (
        args.adl_loss_weight
        * args.pl_loss_weight
        * unsup_adl_loss
    )
    components['sup_adl_loss'] = sup_adl_loss
    components['unsup_adl_loss'] = unsup_adl_loss
    components['total_loss'] = loss
    return loss, components


def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    from util.general_utils import AverageMeter

    head = student[1]
    optimizer, optimizer_hyper = build_optimizers(student, head, args)

    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * (args.batch_size/128) * 1e-3,
    )

    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )

    # inductive
    best_test_acc_lab = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        student.train()
        start = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            data_time = time.perf_counter() - start

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels = class_labels.cuda(non_blocking=True)
            mask_lab = mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out, student_ood, pseudo_out = student(images)
                loss, components = compute_hyp_debgcd_loss(
                    student_proj=student_proj,
                    student_out=student_out,
                    student_ood=student_ood,
                    pseudo_out=pseudo_out,
                    class_labels=class_labels,
                    mask_lab=mask_lab,
                    epoch=epoch,
                    args=args,
                    cluster_criterion=cluster_criterion,
                )

                pstr = (
                    f"cls_loss: {components['cls_loss'].item():.4f} "
                    f"cluster_loss: {components['cluster_loss'].item():.4f} "
                    f"distance sup_con_loss: {components['sup_con_loss_distance'].item():.4f} "
                    f"distance contrastive_loss: {components['contrastive_loss_distance'].item():.4f} "
                    f"angle sup_con_loss: {components['sup_con_loss_angle'].item():.4f} "
                    f"angle contrastive_loss: {components['contrastive_loss_angle'].item():.4f} "
                    f"sup_sdl_loss: {components['sup_sdl_loss'].item():.4f} "
                    f"unsup_sdl_loss: {components['unsup_sdl_loss'].item():.4f} "
                    f"sup_adl_loss: {components['sup_adl_loss'].item():.4f} "
                    f"unsup_adl_loss: {components['unsup_adl_loss'].item():.4f} "
                )

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            optimizer_hyper.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
                optimizer_hyper.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.step(optimizer_hyper)
                fp16_scaler.update()

            whole_time = time.perf_counter() - start
            start = time.perf_counter()
            if batch_idx % args.print_freq == 0:
                args.logger.info(
                    'Epoch: [{}][{}/{}]\t time {:.3f} data_time {:.3f} '
                    'loss {:.3f}\t {}'.format(
                        epoch,
                        batch_idx,
                        len(train_loader),
                        whole_time,
                        data_time,
                        loss.item(),
                        pstr,
                    )
                )

        args.logger.info(
            'Train Epoch: {} Avg Loss: {:.4f} '.format(
                epoch,
                loss_record.avg,
            )
        )

        # Step schedule
        exp_lr_scheduler.step()
        torch.save(student.state_dict(), args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        if epoch:
            args.logger.info(
                'Testing on unlabelled examples in the training data...'
            )
            all_acc, old_acc, new_acc, all_acc2, old_acc2, new_acc2 = test(
                student,
                unlabelled_train_loader,
                epoch=epoch,
                save_name='Train ACC Unlabelled',
                args=args,
            )
            args.logger.info('Testing on disjoint test set...')
            (
                all_acc_test,
                old_acc_test,
                new_acc_test,
                all_acc_test2,
                old_acc_test2,
                new_acc_test2,
            ) = test(
                student,
                test_loader,
                epoch=epoch,
                save_name='Test ACC',
                args=args,
            )

            args.logger.info(
                'Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(
                    all_acc,
                    old_acc,
                    new_acc,
                )
            )
            args.logger.info(
                'Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(
                    all_acc_test,
                    old_acc_test,
                    new_acc_test,
                )
            )

            if old_acc_test > best_test_acc_lab:
                args.logger.info(
                    f'Best ACC on old Classes on disjoint test set: '
                    f'{old_acc_test:.4f}...'
                )
                args.logger.info(
                    'Best Train Accuracies: All {:.4f} | Old {:.4f} | '
                    'New {:.4f}'.format(all_acc, old_acc, new_acc)
                )

                torch.save(
                    student.state_dict(),
                    args.model_path[:-3] + '_best.pt',
                )
                args.logger.info(
                    "model saved to {}.".format(
                        args.model_path[:-3] + '_best.pt'
                    )
                )

                # inductive
                best_test_acc_lab = old_acc_test
                # transductive
                best_train_acc_lab = old_acc
                best_train_acc_ubl = new_acc
                best_train_acc_all = all_acc

                args.logger.info(f'Exp Name: {args.exp_name}')
                args.logger.info(
                    f'Metrics with best model on test set: '
                    f'All: {best_train_acc_all:.4f} '
                    f'Old: {best_train_acc_lab:.4f} '
                    f'New: {best_train_acc_ubl:.4f}'
                )


def test(model, test_loader, epoch, save_name, args):
    from tqdm import tqdm
    from util.cluster_and_log_utils import log_accs_from_preds

    model.eval()

    preds, targets = [], []
    preds_pseudo = []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits, _, logits_pseudo = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            preds_pseudo.append(logits_pseudo.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    preds_pseudo = np.concatenate(preds_pseudo)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, T=epoch, eval_funcs=args.eval_funcs, save_name=save_name, args=args)
    all_acc2, old_acc2, new_acc2 = log_accs_from_preds(y_true=targets, y_pred=preds_pseudo, mask=mask, T=epoch, eval_funcs=args.eval_funcs, save_name=save_name, args=args)
    return all_acc, old_acc, new_acc, all_acc2, old_acc2, new_acc2


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from config import exp_root
    from data.augmentations import get_transform
    from data.get_datasets import get_datasets, get_class_splits
    from models import vision_transformer as vits
    from util.general_utils import init_experiment

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', '--pretrained_path', dest='warmup_model_dir', type=str, default=None,
                        help='Local DINO backbone state_dict; required for training.')
    parser.add_argument('--cars_root', type=str, default=None,
                        help='Stanford Cars root containing devkit/, cars_train/, and cars_test/.')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', '--output_dir', dest='exp_root', type=str, default=exp_root,
                        help='Root directory for experiment logs and default checkpoints.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Optional directory for model.pt; defaults to the experiment log directory.')
    parser.add_argument('--max_train_batches', type=int, default=None,
                        help='Limit optimizer steps per epoch for a smoke test; omit for full training.')
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='simgcd', type=str)
    parser.add_argument('--class_num', default=0, type=int)
    parser.add_argument('--dino', type=str, default='v1')
    parser.add_argument('--seed', default=0, type=int)

    # hyperbolic representation and classification
    parser.add_argument('--c', default=0.05, type=float)
    parser.add_argument('--clip_r', default=0.0, type=float)
    parser.add_argument('--riemannian', action='store_true', default=False)
    parser.add_argument('--hyper_start_epoch', default=0, type=int)
    parser.add_argument('--hyper_end_epoch', default=200, type=int)
    parser.add_argument('--hyper_max_weight', default=1.0, type=float)
    parser.add_argument('--hyper_temp_scale', default=1.0, type=float)

    # auxiliary debiased classifier
    parser.add_argument('--adl_loss_weight', type=float, default=1.0)
    parser.add_argument('--pl_loss_weight', type=float, default=0.5)
    parser.add_argument('--threshold', default=0.95, type=float, help='debiasing threshold')
    parser.add_argument('--pseudo_temp', default=0.1, type=float)
    parser.add_argument('--save_all', action='store_true', default=False)

    # distribution detector
    parser.add_argument('--sdl_loss_weight', type=float, default=0.01)
    parser.add_argument('--num_ood_layers', default=5, type=int)

    # evaluation
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str, default='debgcd_models/dinov1/scars/model.pt')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error('--max_train_batches must be positive.')
    if args.dataset_name == 'scars':
        if args.cars_root is None:
            parser.error('--cars_root is required for Stanford Cars.')
        required_paths = ('devkit/cars_train_annos.mat', 'devkit/cars_test_annos_withlabels.mat',
                          'cars_train', 'cars_test')
        missing_paths = [path for path in required_paths
                         if not os.path.exists(os.path.join(args.cars_root, path))]
        if missing_paths:
            parser.error(f'--cars_root is missing: {", ".join(missing_paths)}')
    if not args.eval_only and args.warmup_model_dir is None:
        parser.error('--pretrained_path is required for training.')
    if args.warmup_model_dir is not None and not os.path.isfile(args.warmup_model_dir):
        parser.error(f'Pretrained weights not found: {args.warmup_model_dir}')
    if args.eval_only and not os.path.isfile(args.eval_path):
        parser.error(f'Evaluation checkpoint not found: {args.eval_path}')
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU is required for this training entrypoint.')
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    if not args.class_num:
        args.num_unlabeled_classes = len(args.unlabeled_classes)
    else:
        args.num_unlabeled_classes = args.class_num - args.num_labeled_classes

    init_experiment(args, runner_name=[f'HypDebGCD_{args.dataset_name}'])
    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.model_dir = args.checkpoint_dir
        args.model_path = os.path.join(args.model_dir, 'model.pt')
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    # Add a handler for stdout and configure it to log to stdout as well
    args.logger.add(sys.stdout)

    # ----------------------
    # SET SEED
    # ----------------------
    set_random_seed(args.seed)

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    # DINO version
    if args.dino == 'v1':
        backbone = vits.__dict__['vit_base']()
    elif args.dino == 'v2':
        from models import vision_transformer2 as vits2
        backbone = vits2.__dict__['vit_base']()
    else:
        raise AttributeError('Unsupported DINO version')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    clip_r = args.clip_r if args.clip_r != 0 else None
    projector = HypDebGCDHead(
        in_dim=args.feat_dim,
        out_dim=args.mlp_out_dim,
        ood_dim=args.num_labeled_classes,
        noodlayers=args.num_ood_layers,
        c=args.c,
        clip_r=clip_r,
        riemannian=args.riemannian,
    )
    model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_path, map_location='cpu'))
        model = model.to(device)
        all_acc, old_acc, new_acc, _, _, _ = test(model, test_loader_unlabelled, epoch=0, save_name='Train ACC Unlabelled', args=args)
        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
    else:
        train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
