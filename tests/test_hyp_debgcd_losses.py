import unittest

import torch
import torch.nn.functional as F

from model import HypSupConLoss, hyp_info_nce_logits


class HypDebGCDLossesTest(unittest.TestCase):
    def test_info_nce_distance_backward(self):
        self._check_info_nce(hyp_c=0.1)

    def test_info_nce_angle_backward(self):
        self._check_info_nce(hyp_c=0)

    def _check_info_nce(self, hyp_c):
        torch.manual_seed(0)
        features = (torch.randn(8, 16) * 0.05).requires_grad_()
        logits, labels = hyp_info_nce_logits(
            features, hyp_c=hyp_c, normalize=False,
        )

        self.assertEqual(logits.shape, (8, 7))
        self.assertEqual(labels.shape, (8,))
        self.assertTrue(torch.isfinite(logits).all().item())

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all().item())

    def test_sup_con_distance_backward(self):
        self._check_sup_con(hyp_c=0.1)

    def test_sup_con_angle_backward(self):
        self._check_sup_con(hyp_c=0)

    def _check_sup_con(self, hyp_c):
        torch.manual_seed(1)
        features = (torch.randn(3, 2, 16) * 0.05).requires_grad_()
        labels = torch.tensor([0, 0, 1])

        loss = HypSupConLoss(hyp_c=hyp_c)(features, labels=labels)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all().item())


if __name__ == '__main__':
    unittest.main()
