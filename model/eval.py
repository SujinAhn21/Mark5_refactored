import os
import sys
import csv
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path:
        sys.path.append(p)

from postprocess_utils import (
    aggregate_segment_probs,
    apply_abstention,
    apply_class_pair_calibration,
    apply_others_calibration,
    apply_temporal_smoothing,
    save_visual_explanation,
)
from vild_config import AudioViLDConfig
from vild_model import LearnableBackgroundEmbedding, ViLDTextHead, build_audio_encoder
from vild_head import DualBranchStudentHead
from vild_parser_teacher import AudioParser
SHARED_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "shared_vild"))
if SHARED_DIR not in sys.path:
    sys.path.append(SHARED_DIR)
from checkpoint_utils import load_checkpoint, resolve_state_dict


def _resolve_test_csv_path():
    candidates = [
        os.path.join(PROJECT_ROOT, "dataset_test.csv"),
        os.path.join(BASE_DIR, "dataset_test.csv"),
        os.path.join(PROJECT_ROOT, "preprocessing", "dataset_test.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"[ERROR] 테스트 인덱스 파일을 찾을 수 없습니다: {candidates}")


def evaluate(mark_version: str):
    config = AudioViLDConfig(mark_version=mark_version)
    device = config.device
    parser = AudioParser(config, segment_mode=True)

    test_files = list(csv.DictReader(open(_resolve_test_csv_path(), newline="", encoding="utf-8")))
    model_path = os.path.join(BASE_DIR, f"student_model_{mark_version}.pth")
    checkpoint = load_checkpoint(model_path, map_location=device)

    encoder = build_audio_encoder(config).to(device)
    encoder.load_state_dict(resolve_state_dict(checkpoint, "model_state_dict", "encoder_state_dict", "model"))
    branch_head = DualBranchStudentHead(config.embedding_dim).to(device)
    branch_state = resolve_state_dict(checkpoint, "branch_state_dict", "head_state_dict", "head")
    if branch_state is not None:
        branch_head.load_state_dict(branch_state, strict=False)
    else:
        print("[WARN] branch_state_dict가 없어 기본 branch head로 평가합니다. 새 모델 재학습이 권장됩니다.")
    text_head = ViLDTextHead(config).to(device)
    text_emb = config.get_class_text_embeddings(for_evaluation=True).to(device)

    # [추가] 학습형 background(others) 임베딩 로드. use_background_embedding=False거나
    # 체크포인트에 없으면(구버전 호환) None으로 두고 아래 max-override 로직을 건너뜀.
    background_embedding = None
    if config.use_background_embedding:
        bg_state = checkpoint.get("background_state_dict")
        if bg_state is not None:
            background_embedding = LearnableBackgroundEmbedding(config.embedding_dim).to(device)
            background_embedding.load_state_dict(bg_state)
            background_embedding.eval()
        else:
            print("[WARN] background_state_dict가 없어 background embedding 없이 평가합니다. 새 모델 재학습이 권장됩니다.")

    encoder.eval()
    branch_head.eval()
    text_head.eval()

    class_names = config.get_classes_for_evaluation()
    label_map = config.get_target_label_map()
    plot_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    all_labels, all_preds, all_probs = [], [], []
    calibration_rows = []
    skipped_label_counter = {}
    for row in test_files:
        path = row["path"]
        label = row["label"]
        if label not in label_map:
            # [추가] 9-class 밖 라벨은 조용히 넘기지 않고 집계
            skipped_label_counter[label] = skipped_label_counter.get(label, 0) + 1
            continue
        segment_records = parser.load_and_segment_with_metadata(path)
        if not segment_records:
            continue

        segment_probs = []
        saliency_scores = [record["saliency"] for record in segment_records]
        with torch.no_grad():
            for record in segment_records:
                seg = record["tensor"]
                if seg.ndim == 3:
                    seg = seg.unsqueeze(0)
                seg = seg.to(device)
                base_features = encoder(seg)
                supervised_features, distill_features = branch_head(base_features)
                sup_logits = text_head(supervised_features, text_emb)
                distill_logits = text_head(distill_features, text_emb)

                if background_embedding is not None:
                    others_idx = class_names.index("others")
                    bg_norm = F.normalize(background_embedding(), dim=0)

                    sup_norm = F.normalize(supervised_features, dim=1)
                    sup_bg_logit = (sup_norm @ bg_norm) / text_head.temperature
                    sup_logits = sup_logits.clone()
                    sup_logits[:, others_idx] = torch.maximum(sup_logits[:, others_idx], sup_bg_logit)

                    distill_norm = F.normalize(distill_features, dim=1)
                    distill_bg_logit = (distill_norm @ bg_norm) / text_head.temperature
                    distill_logits = distill_logits.clone()
                    distill_logits[:, others_idx] = torch.maximum(distill_logits[:, others_idx], distill_bg_logit)

                w = getattr(config, "distill_branch_eval_weight", 0.5)
                prob = (
                    (1 - w) * torch.softmax(sup_logits, dim=-1)
                    + w * torch.softmax(distill_logits, dim=-1)
                ).squeeze(0).cpu().numpy()
                segment_probs.append(prob)

        if config.enable_temporal_smoothing:
            segment_probs = apply_temporal_smoothing(segment_probs, config.temporal_smoothing_alpha)

        aggregated, seg_weights = aggregate_segment_probs(segment_probs, saliency_scores, config)
        calibrated_prob = apply_class_pair_calibration(aggregated, class_names, config)
        calibrated_prob, pred, calib_meta = apply_others_calibration(calibrated_prob, class_names, config)
        calibrated_prob, pred, abstained = apply_abstention(calibrated_prob, class_names, config)
        all_labels.append(label_map[label])
        all_preds.append(pred)
        all_probs.append(calibrated_prob)
        calibration_rows.append({
            "path": path,
            "true_label": label,
            "pred_label": class_names[pred],
            "forced_to_others": calib_meta["forced_to_others"],
            "abstained": abstained,
            "raw_top_conf": calib_meta["raw_top_conf"],
            "raw_margin": calib_meta["raw_margin"],
            "entropy": calib_meta["entropy"],
        })
        save_visual_explanation(path, segment_records, segment_probs, seg_weights, class_names, calibrated_prob, pred, config, plot_dir)

    # [추가] 조용한 탈락 방지: 9-class 밖 라벨 보고 + 평가 샘플 0이면 즉시 중단
    if skipped_label_counter:
        print(
            f"[WARN] 9-class 밖 라벨로 건너뛴 test 파일: {skipped_label_counter} "
            f"(허용 클래스: {sorted(label_map.keys())})"
        )
    if len(all_labels) == 0:
        raise ValueError(
            f"[ERROR] 평가 가능한 test 샘플이 0개입니다. 건너뛴 라벨: {skipped_label_counter}. "
            "dataset_test.csv 라벨이 9-class와 일치하는지 확인하세요."
        )

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 12})
    plt.title(f"Confusion Matrix - {mark_version}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{mark_version}.png"))
    plt.close()

    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(
            os.path.join(plot_dir, f"calibration_details_{mark_version}.csv"),
            index=False,
            encoding="utf-8",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 모델의 성능을 평가합니다.")
    parser.add_argument("--mark_version", type=str, required=True)
    args = parser.parse_args()
    evaluate(mark_version=args.mark_version)
