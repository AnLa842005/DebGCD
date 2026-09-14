import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyptorch.nn import HypLinear, ToPoincare
from model import HypDebGCDHead


class HypDebGCDHeadTest(unittest.TestCase):
    def test_cpu_forward_and_backward(self):
        torch.manual_seed(0)

        batch_size = 4
        feature_dim = 768
        num_seen_classes = 5
        num_classes = 10

        features = torch.randn(batch_size, feature_dim, requires_grad=True)
        head = HypDebGCDHead(
            in_dim=feature_dim,
            out_dim=num_classes,
            ood_dim=num_seen_classes,
            noodlayers=1,
            c=0.05,
            clip_r=2.0,
            riemannian=False,
        )

        representation, logits_gcd, logits_ood, logits_deb = head(features)

        expected_shapes = (
            (representation, (batch_size, feature_dim)),
            (logits_gcd, (batch_size, num_classes)),
            (logits_ood, (batch_size, 2 * num_seen_classes)),
            (logits_deb, (batch_size, num_classes)),
        )
        for output, expected_shape in expected_shapes:
            self.assertEqual(output.shape, expected_shape)
            self.assertEqual(output.device.type, "cpu")
            self.assertTrue(torch.isfinite(output).all().item())

        self.assertIsInstance(head.hyperbolic_projector, ToPoincare)
        self.assertIsInstance(head.last_layer, HypLinear)
        self.assertIsInstance(head.last_layer_deb, HypLinear)

        main_parameter_ids = {id(parameter) for parameter in head.last_layer.parameters()}
        auxiliary_parameter_ids = {id(parameter) for parameter in head.last_layer_deb.parameters()}
        self.assertTrue(main_parameter_ids.isdisjoint(auxiliary_parameter_ids))

        self.assertNotIsInstance(head.last_layer_ood, HypLinear)
        self.assertIsInstance(head.last_layer_ood, nn.Linear)
        for module in head.mlp_ood.modules():
            self.assertNotIsInstance(module, (ToPoincare, HypLinear))
        with torch.no_grad():
            expected_ood = head.last_layer_ood(
                F.normalize(head.mlp_ood(features), dim=-1, p=2)
            )
        torch.testing.assert_close(logits_ood, expected_ood)

        loss = sum(output.square().mean() for output, _ in expected_shapes)
        loss.backward()

        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all().item())
        for name, parameter in head.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)


if __name__ == "__main__":
    unittest.main()
