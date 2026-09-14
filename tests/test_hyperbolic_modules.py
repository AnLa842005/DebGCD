import unittest

import torch

from hyptorch.nn import HypLinear, ToPoincare


class HyperbolicModulesTest(unittest.TestCase):
    def test_cpu_forward_and_backward(self):
        torch.manual_seed(0)

        batch_size = 4
        feature_dim = 768
        num_classes = 10
        curvature = 0.1

        features = torch.randn(batch_size, feature_dim, requires_grad=True)
        projector = ToPoincare(
            c=curvature,
            ball_dim=feature_dim,
            riemannian=False,
            clip_r=2.0,
        )
        classifier = HypLinear(
            in_features=feature_dim,
            out_features=num_classes,
            c=curvature,
        )

        projected = projector(features)
        self.assertEqual(projected.shape, (batch_size, feature_dim))
        self.assertEqual(projected.device.type, "cpu")
        self.assertTrue(torch.isfinite(projected).all().item())

        logits = classifier(projected)
        self.assertEqual(logits.shape, (batch_size, num_classes))
        self.assertEqual(logits.device.type, "cpu")
        self.assertTrue(torch.isfinite(logits).all().item())

        loss = logits.square().mean()
        loss.backward()

        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all().item())
        for parameter in classifier.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())


if __name__ == "__main__":
    unittest.main()
