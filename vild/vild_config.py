# vild_config.py
# 뭐가 많아질수록 힘들구나.  
# [12:41]

import torch
from sentence_transformers import SentenceTransformer
import os

SHARED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared_vild"))
if SHARED_DIR not in os.sys.path:
    os.sys.path.append(SHARED_DIR)

from prompt_bank import get_class_synonyms, get_prompt_templates, get_prompt_texts_for_class

class AudioViLDConfig:
    def __init__(self, mark_version="mark5.0"):
        self.mark_version = mark_version

        # ==============================================================================
        # 1. 클래스 설정 (Mark Version 별)
        # ==============================================================================
        if self.mark_version == "mark4.1":
            self.classes = ["heavy_impact", "others"]
        elif self.mark_version == "mark4.2":
            self.classes = ["dragging", "others"] 
        elif self.mark_version == "mark4.3":
            self.classes = ["construction", "others"] 
        elif self.mark_version == "mark4.4":
            self.classes = ["machine_noise", "others"]
        elif self.mark_version == "mark4.5":
            self.classes = ["media_talking", "others"]
        elif self.mark_version == "mark4.6":
            self.classes = ["water_toilet", "others"]
        elif self.mark_version == "mark4.7":
            self.classes = ["water_shower", "others"]
        elif self.mark_version == "mark4.8":
            self.classes = ["dog_bark", "others"]
        elif self.mark_version == "mark5.0":
            self.classes = [
                "heavy_impact", "dragging", "construction", "machine_noise",
                "media_talking", "water_toilet", "water_shower", "dog_bark",
                "others", 
                "dummy_label" # vild_parser_teacher가 unlabeled 데이터를 처리하기 위한 임시 레이블
            ]
        else:
            raise ValueError(
                f"[Error] Unknown or unsupported mark_version: '{self.mark_version}'.\n"
                f"지원되는 값: ['mark4.1', 'mark4.2', 'mark4.3', 'mark4.4', 'mark4.5', 'mark4.6', 'mark4.7', 'mark4.8', 'mark5.0']"
            )

        # === 기존 속성 유지 ===
        # labeled_classes는 이제 파서가 사용할 모든 클래스를 의미하게 됨
        self.labeled_classes = self.classes
        self.unlabeled_class_identifier = "unlabeled"
        self.num_distinct_labeled_classes = len(self.labeled_classes)


        # ==============================================================================
        # 2. 오디오 및 모델 공통 파라미터
        # ==============================================================================
        # === 오디오 파라미터 ===
        self.sample_rate = 16000
        self.segment_duration = 1.0
        self.segment_samples = int(self.sample_rate * self.segment_duration)
        self.fft_size = 1024
        self.hop_length = 160
        self.n_mels = 64

        # === Segment 단위 처리 ===
        self.segment_length = 101   # Mel spectrogram time frame 수
        self.segment_hop = 50       # Segment 간 stride
        self.max_segments = 5       # Teacher가 사용할 최대 segment 수

        # === 모델 파라미터 ===
        self.embedding_dim = 384
        self.use_background_embedding = True
        self.use_text_aligned_student = True
        self.use_feature_kd = True
        self.feature_kd_weight = 0.3
        self.feature_kd_loss_type = "cosine_l1"
        self.visual_view_type = "mel_delta"
        self.segment_selection_mode = "salient_topk"
        self.max_visual_segments = self.max_segments
        self.logit_temperature = 0.07
        self.segment_aggregation_mode = "confidence_saliency"
        self.segment_confidence_power = 2.0
        self.segment_saliency_power = 1.0
        self.others_confidence_threshold = 0.45
        self.others_margin_threshold = 0.05
        self.others_entropy_threshold = 0.82
        self.class_pair_margin_overrides = {
            ("water_toilet", "water_shower"): 0.03,
            ("construction", "machine_noise"): 0.03,
        }
        self.enable_temporal_smoothing = True
        self.temporal_smoothing_alpha = 0.65
        self.enable_abstention = False
        self.abstention_confidence_threshold = 0.40
        self.explain_topk_segments = 3
        self.save_visual_explanations = True
        self.encoder_type = "cnn"

        # === 학습 파라미터 ===
        self.batch_size = 16
        self.num_epochs = 80
        self.learning_rate = 1e-4
        self.text_loss_weight = 1.0
        self.image_loss_weight = 1.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # === 데이터 경로 ===
        self.audio_dir = os.path.join("data_wav")
        self.prompt_bank_path = os.path.join(SHARED_DIR, "resources", "prompt_bank.json")

        # === 내부 캐시 ===
        self._text_emb = None
        self._eval_text_emb = None # 평가용 텍스트 임베딩 캐시 추가
        self._prompt_texts = None
        self.prompt_templates = get_prompt_templates(self.prompt_bank_path)
        self.class_synonyms = get_class_synonyms(self.prompt_bank_path)


    # ==============================================================================
    # 3. 클래스 및 텍스트 관련 메서드
    # ==============================================================================
    def get_class_index(self, class_name: str) -> int:
        """주어진 클래스 이름의 인덱스를 반환. Unlabeled의 경우 -1을 반환."""
        if class_name in self.labeled_classes:
            return self.labeled_classes.index(class_name)
        elif class_name == self.unlabeled_class_identifier:
            return -1
        else:
            raise ValueError(
                f"[Config Error] '{class_name}'는 mark_version '{self.mark_version}'에 등록되지 않은 클래스입니다.\n"
                f"=> 현재 사용 가능한 클래스: {self.labeled_classes}"
            )

    def get_classes_for_parser(self) -> list:
        """
        데이터 파싱 시 유효한 모든 레이블 목록을 반환. (dummy_label 포함)
        학습 데이터셋 구성 시 사용됨.
        """
        return self.labeled_classes

    def get_classes_for_text_prompts(self) -> list:
        """
        [기존 호환성 유지] 텍스트 프롬프트 생성에 사용될 클래스 목록을 반환.
        기본적으로 파서용 클래스 목록과 동일하게 동작함.
        """
        return self.labeled_classes
    
    def get_classes_for_evaluation(self) -> list:
        """
        [추가된 메서드] 모델 성능 평가에 사용될 실제 타겟 클래스 목록을 반환.
        'dummy_label'과 같이 평가에 사용되지 않는 레이블은 제외됨.
        """
        # self.classes 리스트에서 'dummy_label'을 필터링하여 반환
        return [cls for cls in self.classes if cls != 'dummy_label']

    def get_target_label_map(self) -> dict:
        """
        [수정] 평가용 클래스 목록을 기준으로 라벨-인덱스 맵을 생성함.
        모델의 최종 출력과 매칭시킬 때 사용됨.
        """
        # 평가용 클래스 목록을 사용하도록 변경
        return {class_name: i for i, class_name in enumerate(self.get_classes_for_evaluation())}

    @property
    def num_input_channels(self) -> int:
        if self.visual_view_type == "mel":
            return 1
        if self.visual_view_type == "mel_delta":
            return 3
        if self.visual_view_type == "mel_energy":
            return 2
        return 1

    def get_prompt_texts_for_class(self, class_name: str) -> list:
        return get_prompt_texts_for_class(class_name, self.prompt_bank_path)

    def get_class_text_embeddings(self, for_evaluation: bool = False) -> torch.Tensor:
        """
        클래스 이름에 대한 텍스트 임베딩을 생성하여 반환함.
        :param for_evaluation: True일 경우, 평가용 클래스 목록을 사용하여 임베딩을 생성함.
        """
        if for_evaluation:
            # 평가용 임베딩 생성 및 캐싱
            if self._eval_text_emb is None:
                classes = self.get_classes_for_evaluation()
                model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                aggregated = []
                for class_name in classes:
                    prompts = self.get_prompt_texts_for_class(class_name)
                    emb = model.encode(prompts, convert_to_tensor=True).to(self.device)
                    aggregated.append(emb.mean(dim=0))
                self._eval_text_emb = torch.stack(aggregated, dim=0)
            return self._eval_text_emb
        else:
            # 학습용(기존) 임베딩 생성 및 캐싱
            if self._text_emb is None:
                classes = self.get_classes_for_text_prompts()
                model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                aggregated = []
                for class_name in classes:
                    prompts = self.get_prompt_texts_for_class(class_name)
                    emb = model.encode(prompts, convert_to_tensor=True).to(self.device)
                    aggregated.append(emb.mean(dim=0))
                self._text_emb = torch.stack(aggregated, dim=0)
            return self._text_emb
