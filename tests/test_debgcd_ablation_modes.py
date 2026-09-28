import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from hyptorch.nn import ToPoincare
from model import DebGCDHead, DebGCDModel, HypDebGCDHead
from train_DebGCD import compute_representation_loss


class DebGCDAblationModesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.features = torch.randn(4, 8, requires_grad=True)
        self.class_labels = torch.tensor([0, 0], dtype=torch.long)
        self.mask_lab = torch.tensor([True, True])

    @staticmethod
    def _build_model(use_hyperbolic_head, use_hyperbolic_rep):
        return DebGCDModel(
            backbone=nn.Identity(),
            in_dim=8,
            out_dim=4,
            ood_dim=2,
            nlayers=1,
            noodlayers=1,
            use_hyperbolic_head=use_hyperbolic_head,
            use_hyperbolic_rep=use_hyperbolic_rep,
            c=0.1,
            clip_r=1.2,
            riemannian=False,
        )

    @staticmethod
    def _loss_args(use_hyperbolic_rep):
        return SimpleNamespace(
            use_hyperbolic_rep=use_hyperbolic_rep,
            c=0.1,
            sup_weight=0.35,
            hyper_start_epoch=0,
            hyper_end_epoch=200,
            hyper_max_weight=1.0,
            hyper_temp_scale=0.3,
        )

    def _forward_and_backward(self, model, use_hyperbolic_rep):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        representation, logits_gcd, logits_ood, logits_deb = model(self.features)
        args = self._loss_args(use_hyperbolic_rep)

        if use_hyperbolic_rep:
            loss_rep, log_text = compute_representation_loss(
                representation,
                self.class_labels,
                self.mask_lab,
                epoch=0,
                args=args,
            )
            self.assertIn('distance_contrastive_loss', log_text)
            self.assertIn('angle_contrastive_loss', log_text)
            self.assertIn('lambda_distance: 0.0050', log_text)
        else:
            # The original helper defaults to CUDA because training is GPU-only;
            # keep the production call unchanged while exercising it on CPU here.
            from model import info_nce_logits as real_info_nce_logits

            def cpu_info_nce_logits(features):
                return real_info_nce_logits(features, device='cpu')

            with patch('train_DebGCD.info_nce_logits', cpu_info_nce_logits):
                loss_rep, log_text = compute_representation_loss(
                    representation,
                    self.class_labels,
                    self.mask_lab,
                    epoch=0,
                    args=args,
                )
            self.assertIn('contrastive_loss', log_text)
            self.assertNotIn('distance_contrastive_loss', log_text)

        total_loss = (
            loss_rep
            + logits_gcd.square().mean()
            + logits_ood.square().mean()
            + logits_deb.square().mean()
        )
        self.assertTrue(torch.isfinite(total_loss).item())
        total_loss.backward()
        self.assertIsNotNone(self.features.grad)
        self.assertTrue(torch.isfinite(self.features.grad).all().item())
        optimizer.step()

    def test_e0_uses_euclidean_head_and_euclidean_representation_loss(self):
        model = self._build_model(False, False)

        self.assertIsInstance(model.head, DebGCDHead)
        self.assertNotIsInstance(model.head, HypDebGCDHead)
        self.assertIsNone(model.representation_projector)
        self._forward_and_backward(model, use_hyperbolic_rep=False)

    def test_e1_uses_euclidean_logits_and_hyperbolic_representation_loss(self):
        model = self._build_model(False, True)

        self.assertIsInstance(model.head, DebGCDHead)
        self.assertNotIsInstance(model.head, HypDebGCDHead)
        self.assertIsInstance(model.representation_projector, ToPoincare)
        self.assertFalse(any(
            isinstance(module, HypDebGCDHead) for module in model.modules()
        ))

        representation, logits_gcd, logits_ood, logits_deb = model(self.features)
        direct_outputs = model.head(self.features)
        torch.testing.assert_close(logits_gcd, direct_outputs[1])
        torch.testing.assert_close(logits_ood, direct_outputs[2])
        torch.testing.assert_close(logits_deb, direct_outputs[3])
        torch.testing.assert_close(
            representation,
            model.representation_projector(self.features),
        )

        self._forward_and_backward(model, use_hyperbolic_rep=True)

    def test_e5_uses_hyperbolic_head_and_hyperbolic_representation_loss(self):
        model = self._build_model(True, True)

        self.assertIsInstance(model.head, HypDebGCDHead)
        self.assertIsNone(model.representation_projector)
        self._forward_and_backward(model, use_hyperbolic_rep=True)


if __name__ == '__main__':
    unittest.main()
