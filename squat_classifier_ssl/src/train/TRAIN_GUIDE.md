## 우리 데이터만 사용해서 학습/평가하기

이 문서는 KU-HAR 관련 코드를 제거한 뒤, 우리 데이터만으로 학습·평가·LOSO 등을 수행하는 방법과 파일 구성을 정리합니다.

### 1) 데이터 디렉토리
- 라벨 데이터: `data/manually_labeled/`
- 무라벨(SSL) 데이터(옵션): `data/manually_labeled/ssl/`

폴더 구조는 기존 파이프라인과 동일합니다. 윈도우 CSV 파일은 (길이, 18축) 형태로 로딩됩니다.

---

### 2) 핵심 스크립트
- `scripts/train_from_config.py`
  - 설정 파일(`config/training_config.json`)을 읽어 단일 학습 또는 LOSO 실행.
  - 사용 예:
    - 단일 학습(검증용 대상이 `validation_subject`로 지정된 경우):
      ```bash
      python scripts/train_from_config.py
      ```
    - LOSO 교차검증:
      ```bash
      python scripts/train_from_config.py --loso
      ```
  - XLA/JIT 비활성화, 콜백(ModelCheckpoint/ReduceLROnPlateau/EarlyStopping), 결과 저장 등이 모두 통합되어 있습니다.

- `scripts/loso_with_adapter.py` (옵션)
  - 외부 6축 인코더(예: 자체 제작한 6축 SSL 인코더)가 있을 때 18축 입력에 대해 1x1 Conv 어댑터를 통해 학습·LOSO 평가를 수행.
  - 우리 데이터로 직접 SSL을 해서 18축 인코더를 쓸 경우에는 이 스크립트가 필요 없습니다(어댑터 불필요).
  - 사용 예:
    ```bash
    python scripts/loso_with_adapter.py --encoder-path checkpoints/your_6ch_encoder.keras
    ```

- `scripts/train_with_adapter.py` (옵션)
  - 외부 6축 인코더를 사용해 단일 학습을 수행(어댑터 기반).
  - 우리 데이터 18축 전용 인코더를 사용하는 경우에는 불필요.

---

### 3) 학습/평가에 사용되는 주요 코드
- `src/train/training.py`
  - `TrainingConfig`: 학습 설정(증강, 콜백, 하이퍼파라미터 등)
  - `SquatSequence`: 제너레이터(증강, MixUp/Time CutMix, per-window Z-score 등 포함)
  - `run_training`, `run_loso_cv`: 단일 학습/LOSO 실행 로직(모든 콜백/저장/결과 요약)
  - Optimizer(Adam/AdamW 자동 선택), Focal Loss/Label Smoothing 옵션, freeze→unfreeze 파인튜닝, 클래스 가중치 적용
  - XLA/JIT 비활성화(안정성)

- `src/train/modeling.py`
  - 분류 모델/인코더 정의
  - L2 regularization, `LayerNormalization`, `SpatialDropout1D` 적용

- `src/train/augmentations.py`
  - 시퀀스 증강(노이즈, 스케일링, 타임워프/마스킹, 센서 드롭아웃, 랜덤 시간 쉬프트, Time CutMix 등)

- `src/train/data_utils.py`
  - 데이터 로딩, 윈도우 생성, `list_all_subjects`(LOSO용) 등

- `src/train/ssl_pretrain.py` (옵션)
  - 우리 데이터(무라벨)에 대해 SSL 사전학습을 수행(SimCLR/SimSiam 지원)
  - SimSiam은 배치 크기에 덜 민감하고, 도메인 미스매치에 상대적으로 강함
  - 우리 데이터만으로 사전학습한 인코더를 그대로 분류 모델에 이식 가능(어댑터 불필요)

- `src/train/adapter.py` (옵션)
  - 외부 6축 인코더를 18축 입력에 적용하기 위한 1x1 Conv 어댑터와 분류 헤드
  - 우리 데이터 18축 전용 인코더를 사용할 경우 필요 없음

---

### 4) 설정 파일
- `config/training_config.json` 예시 주요 항목
  - `data_dir`: 라벨 데이터 디렉토리
  - `ssl_dir`: 무라벨 데이터 디렉토리(자체 SSL 시 사용)
  - `target_len`, `batch_size`, `epochs`, `learning_rate`, `patience`
  - `label_smoothing`, `dropout`, `mixup_alpha`, `mixup_prob`, `time_cutmix_prob`, `time_cutmix_alpha`
  - `freeze_epochs`, `use_class_weight`, `use_focal_loss`, `focal_gamma`, `per_window_zscore`, `weight_decay`
  - `optimizer`: `"adam"` 권장(환경에 따라 AdamW 미지원 가능)
  - `validation_subject`: 단일 학습 시 검증으로 쓸 피험자
  - `excluded_subjects`: 특정 피험자를 제외하고 싶을 때 입력(LOSO에서는 fold마다 자동 제외되므로 보통 불필요)

---

### 5) 실행 요약
1. 단일 학습(검증 대상 지정):
   ```bash
   python scripts/train_from_config.py
   ```
2. LOSO 교차검증:
   ```bash
   python scripts/train_from_config.py --loso
   ```
3. (옵션) 외부 6축 인코더 + 어댑터로 LOSO:
   ```bash
   python scripts/loso_with_adapter.py --encoder-path checkpoints/your_6ch_encoder.keras
   ```

---

### 6) 결과물 저장
- 단일 학습: `checkpoints/` 하위에 베스트 가중치(`*.weights.h5`), 학습 이력(`training_history.json`), 곡선(`training_curves.png`) 저장
- LOSO: `checkpoints/loso/subject{n}/...` 형태로 fold별 혼동행렬/리포트/곡선/베스트 가중치/스케일러 저장, `summary.json`에 평균 요약

---

### 7) 참고(안정성)
- CUDA `ptxas` 관련 크래시가 발생하지 않도록 JIT/XLA는 비활성화 되어 있습니다.
- `jit_compile=False`로 모든 `compile()`에 명시.

필요 시, 자체 SSL(우리 데이터만)로 SimSiam 사전학습 → 분류 파이프라인 연동도 바로 설정 가능합니다. 원하시면 해당 경로를 기본으로 전환하는 스크립트/옵션을 추가해 드릴게요.


