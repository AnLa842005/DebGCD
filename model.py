import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hyptorch.nn import HypLinear, ToPoincare
from hyptorch.pmath import dist_matrix
from torch import _weight_norm


class DebGCDHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        ood_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        noodlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        use_hyperbolic_aux=False,
        c=0.1,
        clip_r=1.2,
        riemannian=False,
        use_hyperbolic_main=False,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.use_hyperbolic_main = use_hyperbolic_main
        self.hyperbolic_main_projector = None
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

        # -------------------- debiased classifier --------------------
        self.use_hyperbolic_aux = use_hyperbolic_aux
        self.hyperbolic_aux_projector = None
        self.last_layer_deb = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer_deb.weight_g.data.fill_(1)

        # -------------------- semantic distribution learning --------------------
        if noodlayers:
            noodlayers = max(noodlayers, 1)
            if noodlayers == 1:
                self.mlp_ood = nn.Linear(in_dim, bottleneck_dim)
            elif noodlayers != 0:
                layers_ood = [nn.Linear(in_dim, hidden_dim)]
                if use_bn:
                    layers_ood.append(nn.BatchNorm1d(hidden_dim))
                layers_ood.append(nn.GELU())
                for _ in range(noodlayers - 2):
                    layers_ood.append(nn.Linear(hidden_dim, hidden_dim))
                    if use_bn:
                        layers_ood.append(nn.BatchNorm1d(hidden_dim))
                    layers_ood.append(nn.GELU())
                layers_ood.append(nn.Linear(hidden_dim, bottleneck_dim))
                self.mlp_ood = nn.Sequential(*layers_ood)
        else:
            self.mlp_ood = nn.Identity()

        if noodlayers:
            if noodlayers > 1:
                for m in self.mlp_ood:
                    if isinstance(m, nn.Linear):
                        torch.nn.init.trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            else:
                torch.nn.init.trunc_normal_(self.mlp_ood.weight, std=.02)
                if self.mlp_ood.bias is not None:
                    nn.init.constant_(self.mlp_ood.bias, 0)
        if noodlayers:
            self.last_layer_ood = nn.utils.weight_norm(nn.Linear(bottleneck_dim, 2*ood_dim, bias=False))
        else:
            self.last_layer_ood = nn.utils.weight_norm(nn.Linear(in_dim, 2*ood_dim, bias=False))
        self.last_layer_ood.weight_g.data.fill_(1)

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
            self.last_layer_deb.weight_g.requires_grad = False
            self.last_layer_ood.weight_g.requires_grad = False

        if use_hyperbolic_main:
            # Build all E0 modules first so a fixed seed gives E6 identical aux,
            # representation, and OOD initialization. Only the main classifier
            # is then replaced for the controlled E6 ablation.
            self.hyperbolic_main_projector = ToPoincare(
                c=c,
                ball_dim=in_dim,
                riemannian=riemannian,
                clip_r=clip_r,
            )
            self.last_layer = HypLinear(
                in_features=in_dim,
                out_features=out_dim,
                c=c,
            )

        if use_hyperbolic_aux:
            # Build all E0 modules first so a fixed seed gives E7 identical main,
            # representation, and OOD initialization. Only the auxiliary module
            # is then replaced for the controlled E7 ablation.
            self.hyperbolic_aux_projector = ToPoincare(
                c=c,
                ball_dim=in_dim,
                riemannian=riemannian,
                clip_r=clip_r,
            )
            self.last_layer_deb = HypLinear(
                in_features=in_dim,
                out_features=out_dim,
                c=c,
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        logits_ood = self.last_layer_ood(nn.functional.normalize(self.mlp_ood(x), dim=-1, p=2))
        x_proj = self.mlp(x)
        normalized_x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        if self.use_hyperbolic_main:
            hyp_main_x = self.hyperbolic_main_projector(x)
            logits_gcd = self.last_layer(hyp_main_x)
        else:
            logits_gcd = self.last_layer(normalized_x)
        if self.use_hyperbolic_aux:
            hyp_x = self.hyperbolic_aux_projector(x)
            logits_deb = self.last_layer_deb(hyp_x)
        else:
            logits_deb = self.last_layer_deb(normalized_x)
        return x_proj, logits_gcd, logits_ood, logits_deb


class HypDebGCDHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        ood_dim,
        use_bn=False,
        norm_last_layer=True,
        noodlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        c=0.05,
        clip_r=None,
        riemannian=False,
    ):
        super().__init__()

        self.hyperbolic_projector = ToPoincare(
            c=c,
            ball_dim=in_dim,
            riemannian=riemannian,
            clip_r=clip_r,
        )
        self.last_layer = HypLinear(
            in_features=in_dim,
            out_features=out_dim,
            c=c,
        )

        # -------------------- debiased classifier --------------------
        self.last_layer_deb = HypLinear(
            in_features=in_dim,
            out_features=out_dim,
            c=c,
        )

        # -------------------- semantic distribution learning --------------------
        if noodlayers:
            noodlayers = max(noodlayers, 1)
            if noodlayers == 1:
                self.mlp_ood = nn.Linear(in_dim, bottleneck_dim)
            elif noodlayers != 0:
                layers_ood = [nn.Linear(in_dim, hidden_dim)]
                if use_bn:
                    layers_ood.append(nn.BatchNorm1d(hidden_dim))
                layers_ood.append(nn.GELU())
                for _ in range(noodlayers - 2):
                    layers_ood.append(nn.Linear(hidden_dim, hidden_dim))
                    if use_bn:
                        layers_ood.append(nn.BatchNorm1d(hidden_dim))
                    layers_ood.append(nn.GELU())
                layers_ood.append(nn.Linear(hidden_dim, bottleneck_dim))
                self.mlp_ood = nn.Sequential(*layers_ood)
        else:
            self.mlp_ood = nn.Identity()

        if noodlayers:
            if noodlayers > 1:
                for m in self.mlp_ood:
                    if isinstance(m, nn.Linear):
                        torch.nn.init.trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            else:
                torch.nn.init.trunc_normal_(self.mlp_ood.weight, std=.02)
                if self.mlp_ood.bias is not None:
                    nn.init.constant_(self.mlp_ood.bias, 0)
        if noodlayers:
            self.last_layer_ood = nn.utils.weight_norm(nn.Linear(bottleneck_dim, 2*ood_dim, bias=False))
        else:
            self.last_layer_ood = nn.utils.weight_norm(nn.Linear(in_dim, 2*ood_dim, bias=False))
        self.last_layer_ood.weight_g.data.fill_(1)

        if norm_last_layer:
            self.last_layer_ood.weight_g.requires_grad = False

    def forward(self, x):
        logits_ood = self.last_layer_ood(nn.functional.normalize(self.mlp_ood(x), dim=-1, p=2))
        hyp_x = self.hyperbolic_projector(x)
        logits_gcd = self.last_layer(hyp_x)
        logits_deb = self.last_layer_deb(hyp_x)
        return hyp_x, logits_gcd, logits_ood, logits_deb


class EuclideanRepresentationHead(nn.Module):
    """Original DebGCD MLP used when only the classifier head is hyperbolic."""

    def __init__(self, in_dim, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class DebGCDModel(nn.Sequential):
    """Compose independent classifier-head and representation-loss geometry."""

    def __init__(
        self,
        backbone,
        in_dim,
        out_dim,
        ood_dim,
        nlayers=3,
        noodlayers=3,
        use_hyperbolic_head=False,
        use_hyperbolic_rep=False,
        c=0.1,
        clip_r=1.2,
        riemannian=False,
        use_hyperbolic_aux_only=False,
        use_hyperbolic_main_only=False,
    ):
        if use_hyperbolic_aux_only and (
            use_hyperbolic_head
            or use_hyperbolic_rep
            or use_hyperbolic_main_only
        ):
            raise ValueError(
                'use_hyperbolic_aux_only cannot be combined with '
                'other hyperbolic modes.'
            )
        if use_hyperbolic_main_only and (
            use_hyperbolic_head or use_hyperbolic_rep
        ):
            raise ValueError(
                'use_hyperbolic_main_only cannot be combined with '
                'use_hyperbolic_head or use_hyperbolic_rep.'
            )

        if use_hyperbolic_head:
            head = HypDebGCDHead(
                in_dim=in_dim,
                out_dim=out_dim,
                ood_dim=ood_dim,
                noodlayers=noodlayers,
                c=c,
                clip_r=clip_r,
                riemannian=riemannian,
            )
        else:
            head = DebGCDHead(
                in_dim=in_dim,
                out_dim=out_dim,
                ood_dim=ood_dim,
                nlayers=nlayers,
                noodlayers=noodlayers,
                use_hyperbolic_aux=use_hyperbolic_aux_only,
                c=c,
                clip_r=clip_r,
                riemannian=riemannian,
                use_hyperbolic_main=use_hyperbolic_main_only,
            )

        super().__init__(backbone, head)
        self.use_hyperbolic_head = use_hyperbolic_head
        self.use_hyperbolic_rep = use_hyperbolic_rep
        self.use_hyperbolic_aux_only = use_hyperbolic_aux_only
        self.use_hyperbolic_main_only = use_hyperbolic_main_only

        if use_hyperbolic_rep and not use_hyperbolic_head:
            self.representation_projector = ToPoincare(
                c=c,
                ball_dim=in_dim,
                riemannian=riemannian,
                clip_r=clip_r,
            )
        elif use_hyperbolic_head and not use_hyperbolic_rep:
            self.representation_projector = EuclideanRepresentationHead(
                in_dim=in_dim,
                nlayers=nlayers,
            )
        else:
            self.representation_projector = None

    @property
    def head(self):
        return self._modules['1']

    def forward(self, x):
        features = self._modules['0'](x)
        representation, logits_gcd, logits_ood, logits_deb = self.head(features)
        if self.representation_projector is not None:
            representation = self.representation_projector(features)
        return representation, logits_gcd, logits_ood, logits_deb


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
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

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class HypSupConLoss(torch.nn.Module):
    """HypCD supervised contrastive loss for distance and angle similarities."""

    def __init__(self, temperature=0.07, contrast_mode='all', hyp_c=0):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.hyp_c = hyp_c

    def forward(self, features, labels=None, mask=None):
        device = features.device
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
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
            similarity = torch.matmul(
                F.normalize(anchor_feature, dim=-1, p=2),
                F.normalize(contrast_feature, dim=-1, p=2).T,
            )
        else:
            similarity = -dist_matrix(anchor_feature, contrast_feature, c=self.hyp_c)
        anchor_dot_contrast = similarity / self.temperature

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # HypCD uses no temperature/base-temperature multiplier here.
        loss = -mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def hyp_info_nce_logits(features, n_views=2, temperature=1.0, hyp_c=0, normalize=True):
    """HypCD InfoNCE logits using Poincare distance or normalized angles."""
    batch_size = features.size(0) // n_views
    labels = torch.arange(batch_size, device=features.device).repeat(n_views)
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

    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)
    return logits / temperature, labels


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
