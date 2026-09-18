import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from model import DistillLoss, HypDebGCDHead
from train_HypDebGCD import (
    assert_optimizer_parameter_sets_disjoint,
    build_optimizers,
    compute_hyp_debgcd_loss,
    optimizer_parameter_ids,
)


class HypDebGCDTrainingStepTest(unittest.TestCase):
    def test_complete_cpu_training_step(self):
        torch.manual_seed(0)

        batch_size = 2
        feature_dim = 768
        num_seen_classes = 5
        num_classes = 10

        backbone = nn.Linear(feature_dim, feature_dim)
        head = HypDebGCDHead(
            in_dim=feature_dim,
            out_dim=num_classes,
            ood_dim=num_seen_classes,
            noodlayers=1,
            c=0.05,
            clip_r=2.0,
            riemannian=False,
        )
        student = nn.Sequential(backbone, head)

        args = SimpleNamespace(
            lr=0.1,
            batch_size=batch_size,
            momentum=0.9,
            weight_decay=1e-4,
            sup_weight=0.35,
            memax_weight=2.0,
            c=0.05,
            hyper_start_epoch=0,
            hyper_end_epoch=2,
            hyper_max_weight=1.0,
            hyper_temp_scale=1.0,
            sdl_loss_weight=0.01,
            adl_loss_weight=1.0,
            pl_loss_weight=0.5,
            threshold=0.0,
            pseudo_temp=0.1,
        )

        optimizer, optimizer_hyper = build_optimizers(student, head, args)
        assert_optimizer_parameter_sets_disjoint(
            optimizer,
            optimizer_hyper,
        )

        sgd_parameter_ids = optimizer_parameter_ids(optimizer)
        hyperbolic_parameter_ids = optimizer_parameter_ids(optimizer_hyper)
        self.assertTrue(
            sgd_parameter_ids.isdisjoint(hyperbolic_parameter_ids)
        )

        expected_hyperbolic_ids = {
            id(parameter)
            for module in (head.last_layer, head.last_layer_deb)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(hyperbolic_parameter_ids, expected_hyperbolic_ids)

        expected_sgd_ids = {
            id(parameter)
            for module in (backbone, head.mlp_ood, head.last_layer_ood)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(sgd_parameter_ids, expected_sgd_ids)

        inputs = torch.randn(
            2 * batch_size,
            feature_dim,
            requires_grad=True,
        )
        class_labels = torch.tensor([0, 1], dtype=torch.long)
        mask_lab = torch.tensor([True, False])

        student_proj, student_out, student_ood, pseudo_out = student(inputs)
        outputs = (student_proj, student_out, student_ood, pseudo_out)
        for output in outputs:
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all().item())

        cluster_criterion = DistillLoss(
            warmup_teacher_temp_epochs=1,
            nepochs=2,
            ncrops=2,
            warmup_teacher_temp=0.07,
            teacher_temp=0.04,
        )
        total_loss, loss_components = compute_hyp_debgcd_loss(
            student_proj=student_proj,
            student_out=student_out,
            student_ood=student_ood,
            pseudo_out=pseudo_out,
            class_labels=class_labels,
            mask_lab=mask_lab,
            epoch=0,
            args=args,
            cluster_criterion=cluster_criterion,
        )

        for name, component in loss_components.items():
            self.assertTrue(
                torch.isfinite(component).all().item(),
                name,
            )
        self.assertTrue(torch.isfinite(total_loss).all().item())

        optimizer.zero_grad()
        optimizer_hyper.zero_grad()
        total_loss.backward()

        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all().item())
        for name, parameter in student.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(
                    torch.isfinite(parameter.grad).all().item(),
                    name,
                )

        optimizer.step()
        optimizer_hyper.step()


if __name__ == '__main__':
    unittest.main()
