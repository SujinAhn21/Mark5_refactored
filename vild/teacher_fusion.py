from dataclasses import dataclass

import torch


@dataclass
class TeacherFusionOutput:
    logits: torch.Tensor
    features: torch.Tensor
    specialist_weights: dict
    disagreement: torch.Tensor
    uncertainty: torch.Tensor


class WeightedTeacherFusion:
    def __init__(self, student_class_map, embedding_dim, device):
        self.student_class_map = student_class_map
        self.embedding_dim = embedding_dim
        self.device = device

    def fuse(self, specialist_outputs):
        batch_size = next(iter(specialist_outputs.values()))["logits"].shape[0]
        num_classes = len(self.student_class_map)
        ensembled_logits = torch.zeros(batch_size, num_classes, device=self.device)
        others_logits_sum = torch.zeros(batch_size, 1, device=self.device)
        feature_sum = torch.zeros(batch_size, self.embedding_dim, device=self.device)
        weight_sum = torch.zeros(batch_size, 1, device=self.device)
        specialist_weights = {}

        for class_name, output in specialist_outputs.items():
            logits = output["logits"]
            features = output["features"]
            probs = torch.softmax(logits, dim=1)
            positive_weight = probs[:, 0:1]
            specialist_weights[class_name] = positive_weight.detach().cpu()

            target_idx = self.student_class_map[class_name]
            ensembled_logits[:, target_idx] = logits[:, 0]
            others_logits_sum += logits[:, 1:2]
            feature_sum += features * positive_weight
            weight_sum += positive_weight

        others_idx = self.student_class_map["others"]
        ensembled_logits[:, others_idx] = (others_logits_sum / len(specialist_outputs)).squeeze(1)
        fused_features = feature_sum / weight_sum.clamp_min(1e-6)
        stacked_weights = torch.stack(
            [specialist_weights[name].to(self.device).squeeze(1) for name in specialist_outputs.keys()],
            dim=1,
        )
        top2 = torch.topk(stacked_weights, k=min(2, stacked_weights.shape[1]), dim=1).values
        if top2.shape[1] == 1:
            disagreement = torch.zeros(batch_size, device=self.device)
        else:
            disagreement = top2[:, 0] - top2[:, 1]
        uncertainty = 1.0 - stacked_weights.max(dim=1).values
        return TeacherFusionOutput(
            logits=ensembled_logits,
            features=fused_features,
            specialist_weights=specialist_weights,
            disagreement=disagreement,
            uncertainty=uncertainty,
        )
