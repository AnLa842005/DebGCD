import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyptorch.nn import HypLinear, ToPoincare
from model import DebGCDHead, DebGCDModel, SupConLoss
from train_DebGCD import compute_pseudo_labels, compute_representation_loss


class DebGCDE7Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.features = torch.randn(4, 8)
        self.class_labels = torch.tensor([0, 1], dtype=torch.long)
        self.mask_lab = torch.tensor([True, False])

    @staticmethod
    def _build_model(use_hyperbolic_aux_only=True):
        return DebGCDModel(
            backbone=nn.Identity(),
            in_dim=8,
            out_dim=4,
            ood_dim=2,
            nlayers=1,
            noodlayers=1,
            use_hyperbolic_head=False,
            use_hyperbolic_rep=False,
            use_hyperbolic_aux_only=use_hyperbolic_aux_only,
            c=0.1,
            clip_r=1.2,
            riemannian=False,
        )

    @staticmethod
    def _loss_args():
        return SimpleNamespace(
            use_hyperbolic_rep=False,
            c=0.1,
            sup_weight=0.35,
            hyper_start_epoch=0,
            hyper_end_epoch=200,
            hyper_max_weight=1.0,
            hyper_temp_scale=0.3,
        )

    def test_e7_routes_each_output_through_the_required_geometry(self):
        model = self._build_model()
        head = model.head

        self.assertIsInstance(head, DebGCDHead)
        self.assertIsNone(model.representation_projector)
        self.assertIsInstance(head.last_layer, nn.Linear)
        self.assertIsNone(head.last_layer.bias)
        self.assertIsInstance(head.hyperbolic_aux_projector, ToPoincare)
        self.assertIsInstance(head.last_layer_deb, HypLinear)
        self.assertIsInstance(head.last_layer_ood, nn.Linear)

        main_parameter_ids = {id(parameter) for parameter in head.last_layer.parameters()}
        aux_parameter_ids = {id(parameter) for parameter in head.last_layer_deb.parameters()}
        self.assertTrue(main_parameter_ids.isdisjoint(aux_parameter_ids))

        representation, logits_gcd, logits_ood, logits_deb = model(self.features)
        with torch.no_grad():
            expected_representation = head.mlp(self.features)
            expected_gcd = head.last_layer(F.normalize(self.features, dim=-1, p=2))
            expected_deb = head.last_layer_deb(
                head.hyperbolic_aux_projector(self.features)
            )
            expected_ood = head.last_layer_ood(
                F.normalize(head.mlp_ood(self.features), dim=-1, p=2)
            )

        torch.testing.assert_close(representation, expected_representation)
        torch.testing.assert_close(logits_gcd, expected_gcd)
        torch.testing.assert_close(logits_deb, expected_deb)
        torch.testing.assert_close(logits_ood, expected_ood)

    def test_e7_preserves_e0_initialization_outside_the_auxiliary_classifier(self):
        torch.manual_seed(123)
        e0_model = self._build_model(use_hyperbolic_aux_only=False)
        torch.manual_seed(123)
        e7_model = self._build_model(use_hyperbolic_aux_only=True)

        for e0_module, e7_module in (
            (e0_model.head.mlp, e7_model.head.mlp),
            (e0_model.head.last_layer, e7_model.head.last_layer),
            (e0_model.head.mlp_ood, e7_model.head.mlp_ood),
            (e0_model.head.last_layer_ood, e7_model.head.last_layer_ood),
        ):
            e0_state = e0_module.state_dict()
            e7_state = e7_module.state_dict()
            self.assertEqual(e0_state.keys(), e7_state.keys())
            for name in e0_state:
                torch.testing.assert_close(e7_state[name], e0_state[name])

    def test_e7_uses_euclidean_representation_losses(self):
        model = self._build_model()
        representation, _, _, _ = model(self.features)

        from model import info_nce_logits as real_info_nce_logits

        def cpu_info_nce_logits(features):
            return real_info_nce_logits(features, device='cpu')

        with patch(
            'train_DebGCD.info_nce_logits',
            side_effect=cpu_info_nce_logits,
        ) as info_nce_mock:
            with patch('train_DebGCD.SupConLoss', wraps=SupConLoss) as sup_con_mock:
                with patch(
                    'train_DebGCD.hyp_info_nce_logits',
                    side_effect=AssertionError('E7 must not use hyp_info_nce_logits'),
                ), patch(
                    'train_DebGCD.HypSupConLoss',
                    side_effect=AssertionError('E7 must not use HypSupConLoss'),
                ):
                    loss_rep, log_text = compute_representation_loss(
                        representation,
                        self.class_labels,
                        self.mask_lab,
                        epoch=0,
                        args=self._loss_args(),
                    )

        self.assertTrue(torch.isfinite(loss_rep).item())
        info_nce_mock.assert_called_once()
        sup_con_mock.assert_called_once()
        self.assertIn('sup_con_loss', log_text)
        self.assertIn('contrastive_loss', log_text)
        self.assertNotIn('distance_contrastive_loss', log_text)
        self.assertNotIn('lambda_distance', log_text)

    def test_e7_pseudo_labels_come_from_main_and_adl_updates_aux(self):
        model = self._build_model()
        inputs = self.features.clone().requires_grad_(True)
        _, student_out, _, pseudo_out = model(inputs)

        max_probs, targets_u, accepted_mask = compute_pseudo_labels(
            student_out,
            self.mask_lab,
            threshold=0.0,
        )
        expected_main_logits = torch.cat(
            [features[~self.mask_lab] for features in (student_out / 0.1).chunk(2)],
            dim=0,
        )
        expected_targets = expected_main_logits.detach().softmax(dim=-1).argmax(dim=-1)
        torch.testing.assert_close(targets_u, expected_targets)
        self.assertFalse(max_probs.requires_grad)
        self.assertFalse(targets_u.requires_grad)
        self.assertFalse(accepted_mask.requires_grad)

        unsup_aux_logits = torch.cat(
            [features[~self.mask_lab] for features in (pseudo_out / 0.1).chunk(2)],
            dim=0,
        )
        unsup_adl_loss = (
            F.cross_entropy(unsup_aux_logits, targets_u, reduction='none')
            * accepted_mask
            * torch.ones_like(max_probs)
        ).mean()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        unsup_adl_loss.backward()

        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.head.last_layer_deb.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.head.last_layer.parameters()
        ))
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all().item())
        optimizer.step()

    def test_e7_checkpoint_round_trip(self):
        model = self._build_model()
        outputs = model(self.features)
        loss = sum(output.square().mean() for output in outputs)
        loss.backward()

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_path = f'{checkpoint_dir}/model.pt'
            torch.save(model.state_dict(), checkpoint_path)
            reloaded_model = self._build_model()
            reloaded_model.load_state_dict(
                torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            )
            reloaded_outputs = reloaded_model(self.features)

        for output, reloaded_output in zip(outputs, reloaded_outputs):
            torch.testing.assert_close(output, reloaded_output)

    def test_aux_only_mode_rejects_other_hyperbolic_modes(self):
        with self.assertRaises(ValueError):
            DebGCDModel(
                backbone=nn.Identity(),
                in_dim=8,
                out_dim=4,
                ood_dim=2,
                use_hyperbolic_aux_only=True,
                use_hyperbolic_rep=True,
            )


if __name__ == '__main__':
    unittest.main()
