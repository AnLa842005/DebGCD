import math
import unittest

import torch

from train_DebGCD import PseudoLabelDiagnostics


class PseudoLabelDiagnosticsTest(unittest.TestCase):
    def test_correctness_and_activity_statistics(self):
        diagnostics = PseudoLabelDiagnostics(num_labeled_classes=2)

        confidence = torch.tensor([0.90, 0.60, 0.80, 0.95, 0.40, 0.75])
        predictions = torch.tensor([0, 2, 1, 3, 3, 1])
        accepted = torch.tensor([True, False, True, True, False, True])
        ground_truth = torch.tensor([0, 2, 3, 3, 1, 1])
        ood_certainty = torch.tensor([0.8, 0.2, 0.5, 1.0, 0.4, 0.6])

        diagnostics.update(
            max_probs=confidence[:2],
            targets_u=predictions[:2],
            accepted_mask=accepted[:2],
            ground_truth=ground_truth[:2],
            ood_certainty=ood_certainty[:2],
            unsup_adl_loss=torch.tensor(0.3, requires_grad=True),
        )
        diagnostics.update(
            max_probs=confidence[2:],
            targets_u=predictions[2:],
            accepted_mask=accepted[2:],
            ground_truth=ground_truth[2:],
            ood_certainty=ood_certainty[2:],
            unsup_adl_loss=torch.tensor(0.0, requires_grad=True),
        )

        summary = diagnostics.summary()

        self.assertEqual(summary['total_views'], 6)
        self.assertEqual(summary['accepted_views'], 4)
        self.assertAlmostEqual(summary['acceptance_rate'], 4 / 6)
        self.assertEqual(summary['predicted_old'], 3)
        self.assertEqual(summary['predicted_new'], 3)
        self.assertEqual(summary['accepted_predicted_old'], 3)
        self.assertEqual(summary['accepted_predicted_new'], 1)
        self.assertAlmostEqual(summary['pseudo_label_accuracy'], 4 / 6)
        self.assertAlmostEqual(summary['accepted_pseudo_label_accuracy'], 3 / 4)
        self.assertAlmostEqual(summary['old_pseudo_label_accuracy'], 2 / 3)
        self.assertAlmostEqual(summary['new_pseudo_label_accuracy'], 2 / 3)
        self.assertAlmostEqual(summary['accepted_old_pseudo_label_accuracy'], 1.0)
        self.assertAlmostEqual(summary['accepted_new_pseudo_label_accuracy'], 1 / 2)
        self.assertAlmostEqual(summary['ood_certainty_mean'], 3.5 / 6)
        self.assertAlmostEqual(summary['accepted_ood_certainty_mean'], 2.9 / 4)
        self.assertAlmostEqual(summary['unsup_adl_loss_mean'], 0.1)
        self.assertEqual(summary['active_batches'], 1)
        self.assertEqual(summary['total_batches'], 2)
        self.assertAlmostEqual(summary['active_batch_rate'], 0.5)
        self.assertAlmostEqual(summary['confidence_mean'], confidence.mean().item())
        self.assertAlmostEqual(summary['confidence_min'], 0.4)
        self.assertAlmostEqual(summary['confidence_max'], 0.95)

    def test_empty_accepted_subset_is_reported_as_nan(self):
        diagnostics = PseudoLabelDiagnostics(num_labeled_classes=2)
        diagnostics.update(
            max_probs=torch.tensor([0.4, 0.5]),
            targets_u=torch.tensor([0, 2]),
            accepted_mask=torch.tensor([False, False]),
            ground_truth=torch.tensor([0, 3]),
            ood_certainty=torch.tensor([0.3, 0.7]),
            unsup_adl_loss=torch.tensor(0.0),
        )

        summary = diagnostics.summary()

        self.assertTrue(math.isnan(summary['accepted_pseudo_label_accuracy']))
        self.assertTrue(math.isnan(summary['accepted_old_pseudo_label_accuracy']))
        self.assertTrue(math.isnan(summary['accepted_new_pseudo_label_accuracy']))
        self.assertTrue(math.isnan(summary['accepted_ood_certainty_mean']))


if __name__ == '__main__':
    unittest.main()
