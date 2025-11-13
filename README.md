# Squat Classifier SSL

실시간 스쿼트 자세 분류 시스템 - IMU 센서 기반 딥러닝 분류기

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
- [설치](#설치)
- [사용법](#사용법)
  - [모델 학습](#모델-학습)
  - [실시간 추론](#실시간-추론)
- [데이터 구조](#데이터-구조)
- [설정 파일](#설정-파일)
- [문서](#문서)
- [기여](#기여)
- [라이선스](#라이선스)

## 🎯 개요

이 프로젝트는 3개의 IMU (Inertial Measurement Unit) 센서 데이터를 사용하여 실시간으로 스쿼트 자세를 분류하는 딥러닝 시스템입니다. 허리, 허벅지, 종아리에 부착된 센서로부터 수집된 가속도 및 자이로스코프 데이터를 분석하여 5가지 자세 클래스를 분류합니다.

### 분류 클래스

- **Correct (정자세)**: 올바른 스쿼트 자세
- **Knee Valgus (무릎 모임)**: 무릎이 안쪽으로 모이는 자세
- **Butt Wink (벗 윙크)**: 엉덩이가 말리는 자세
- **Excessive Lean (상체 과다 숙임)**: 상체가 과도하게 앞으로 기울어지는 자세
- **Partial Squat (얕은 스쿼트)**: 깊이가 부족한 스쿼트

## ✨ 주요 기능

### 1. 실시간 추론
- **슬라이딩 윈도우 기반 추론**: 512 샘플 윈도우, 0.25-0.5초 stride
- **Rep 단위 감지**: FSM (Finite State Machine) 기반 1-rep 스쿼트 자동 감지
- **스무딩 파이프라인**: EMA, 불확실성 보류, 다수결 투표를 통한 안정적인 분류
- **리샘플링 추론**: Rep 완료 시 전체 구간을 리샘플링하여 최종 분류

### 2. 모델 학습
- **SSL (Self-Supervised Learning) 지원**: SimCLR, SimSiam 등
- **데이터 증강**: Mixup, Time CutMix, 다양한 시계열 증강 기법
- **LOSO (Leave-One-Subject-Out) 교차 검증**: 피험자별 일반화 성능 평가
- **Focal Loss**: 클래스 불균형 문제 해결

### 3. 카메라 통합
- **MediaPipe 기반 자세 추정**: 카메라 피드에서 자세 분류
- **실시간 오버레이**: 분류 결과를 카메라 화면에 실시간 표시
- **Rep 클립 저장**: 각 스쿼트 rep의 비디오 클립 자동 저장 및 재생

### 4. 센서 데이터 처리
- **3개 IMU 센서**: 허리(s0), 허벅지(s1), 종아리(s2)
- **18차원 특징**: 각 센서당 가속도 3축 + 자이로 3축
- **110 Hz 샘플링 레이트**: 고주파 센서 데이터 수집

## 📁 프로젝트 구조

```
squat_classifier_ssl/
├── src/                    # 소스 코드
│   ├── train/              # 학습 관련 모듈
│   │   ├── modeling.py     # 모델 아키텍처
│   │   ├── training.py     # 학습 파이프라인
│   │   ├── data_utils.py   # 데이터 로딩 및 전처리
│   │   ├── augmentations.py # 데이터 증강
│   │   ├── ssl_pretrain.py # SSL 사전 학습
│   │   └── totflite.py     # TFLite 변환
│   ├── ssl/                # SSL 관련 모듈
│   └── utils/              # 유틸리티 함수
├── scripts/                # 실행 스크립트
│   ├── realtime_new_pipeline.py  # 실시간 추론 메인 스크립트
│   └── train_from_config.py      # 설정 파일 기반 학습
├── notebooks/                # 실험 및 분석용 Jupyter 노트북
├── config/                 # 설정 파일
│   └── training_config.json
├── data/                   # 데이터 디렉토리
│   ├── manually_labeled/   # 수동 라벨링된 데이터
│   └── raw/               # 원시 데이터
├── checkpoints/           # 학습된 모델 체크포인트
├── exports/               # TFLite 모델 및 스케일러
├── logs/                  # 로그 및 세션 기록
│   └── realtime_sessions/ # 실시간 추론 세션 로그
└── docs/                  # 문서
    └── realtime_inference_pipeline.md
```

## 🚀 설치

### 요구사항

- Python 3.x
- TensorFlow 2.x
- NumPy, Pandas, Scikit-learn
- OpenCV (카메라 기능 사용 시)
- MediaPipe (카메라 기능 사용 시)
- Picamera2 (라즈베리파이 카메라 사용 시)

### 설치 방법

1. **저장소 클론**
```bash
git clone https://github.com/choijaeh01/Squat-Realtime-Classifier.git
cd squat_classifier_ssl
```

2. **Conda 환경 생성 (선택사항)**
```bash
conda env create -f environment.yml
conda activate squat_ssl
```

3. **필수 패키지 설치**
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib opencv-python mediapipe
```

4. **ESP32 센서 설정** (실시간 추론 사용 시)
   - ESP32 보드에 3개의 IMU 센서 연결
   - Wi-Fi 네트워크 설정
   - UDP 패킷 전송 설정 (포트 12345)

## 📖 사용법

### 모델 학습

#### 1. 설정 파일 준비

`config/training_config.json` 파일을 수정하여 학습 설정을 지정합니다:

```json
{
  "data_dir": "data/manually_labeled",
  "ssl_dir": "data/manually_labeled/ssl",
  "output_dir": "checkpoints",
  "target_len": 512,
  "batch_size": 64,
  "epochs": 120,
  "learning_rate": 0.0001,
  "use_ssl": false,
  "use_focal_loss": true,
  "per_window_zscore": true
}
```

#### 2. 학습 실행

```bash
python scripts/train_from_config.py --config config/training_config.json
```

#### 3. SSL 사전 학습 (선택사항)

```json
{
  "use_ssl": true,
  "ssl_epochs": 100
}
```

#### 4. TFLite 변환

학습 완료 후 모델을 TFLite 형식으로 변환:

```python
from src.train.totflite import convert_to_tflite
convert_to_tflite(
    keras_model_path="checkpoints/squat_classifier_best.weights.h5",
    output_path="exports/squat_classifier_fp16.tflite"
)
```

### 실시간 추론

#### 기본 실행

```bash
python scripts/realtime_new_pipeline.py \
    --model exports/squat_classifier_fp16.tflite \
    --scaler checkpoints/squat_scaler_18axis.pkl
```

#### 카메라 연계 실행

```bash
python scripts/realtime_new_pipeline.py \
    --model exports/squat_classifier_fp16.tflite \
    --scaler checkpoints/squat_scaler_18axis.pkl \
    --enable-camera \
    --camera-display
```

#### 주요 옵션

- `--model`: TFLite 모델 경로
- `--scaler`: StandardScaler 피클 파일 경로
- `--enable-camera`: 카메라 기능 활성화
- `--camera-display`: 카메라 화면 표시
- `--sliding-stride-sec`: 슬라이딩 윈도우 stride (기본값: 0.5초)
- `--ema-alpha`: EMA 스무딩 계수 (기본값: 0.3)
- `--uncertainty-p-max`: 불확실성 임계값 (기본값: 0.25)

#### 키보드 단축키

- `q`: 프로그램 종료
- `s`: 카메라 녹화 시작
- `r`: 최근 rep 클립 재생 (0.5배속)

## 📊 데이터 구조

### 수동 라벨링 데이터

```
data/manually_labeled/
├── class0/          # Correct
├── class1/          # Knee Valgus
├── class2/          # Butt Wink
├── class3/          # Excessive Lean
├── class4/          # Partial Squat
└── ssl/             # SSL용 비라벨 데이터
```

각 클래스 폴더에는 CSV 파일이 포함되어 있으며, 형식은 다음과 같습니다:

```csv
millis,ax,ay,az,gx,gy,gz,...
```

### 원시 데이터

```
data/raw/
├── labeled/         # 라벨링된 원시 데이터
└── unlabeled/      # 비라벨 원시 데이터
```

## ⚙️ 설정 파일

### training_config.json

주요 설정 항목:

- **data_dir**: 학습 데이터 디렉토리
- **target_len**: 시퀀스 길이 (기본값: 512)
- **batch_size**: 배치 크기
- **epochs**: 학습 에포크 수
- **learning_rate**: 학습률
- **use_ssl**: SSL 사용 여부
- **use_focal_loss**: Focal Loss 사용 여부
- **per_window_zscore**: Per-window Z-score 정규화 사용 여부
- **dropout**: Dropout 비율
- **mixup_alpha**: Mixup 증강 파라미터
- **time_cutmix_prob**: Time CutMix 확률

자세한 내용은 `config/training_config.json` 파일을 참조하세요.

## 📚 문서

- [실시간 추론 파이프라인 상세 설명](docs/realtime_inference_pipeline.md)
- [학습 가이드](src/train/TRAIN_GUIDE.md)

## 🔧 주요 기술 스택

- **딥러닝 프레임워크**: TensorFlow/Keras
- **모델 아키텍처**: 1D ResNet 기반 시계열 분류기
- **추론 백엔드**: TensorFlow Lite (모바일/엣지 디바이스 최적화)
- **데이터 처리**: NumPy, Pandas
- **컴퓨터 비전**: OpenCV, MediaPipe
- **하드웨어**: ESP32, IMU 센서 (MPU6050 또는 유사)

## 🎓 알고리즘 개요

### Rep 감지 알고리즘

1. **s1_gz 기반 FSM**: 허벅지 센서의 각속도 Z축을 사용한 상태 머신
   - Idle → Descent → Bottom → Ascent → Idle
2. **s0_gy 검증**: 허리 센서의 각속도 Y축으로 실제 스쿼트 동작 검증
   - Rep 시작 후 1초 내 s0_gy가 0.5 이상 증가해야 유효한 rep으로 인정

### 스무딩 파이프라인

1. **EMA (Exponential Moving Average)**: 확률 스무딩
2. **불확실성 보류**: 낮은 신뢰도 구간은 TRANSITION으로 표시
3. **다수결 투표**: 최근 k개 창의 분류 결과 투표

### 전처리

1. **StandardScaler 변환**: 학습 시 사용한 스케일러 적용
2. **Per-window Z-score**: 각 윈도우 내 독립적 정규화
3. **±6σ 클리핑**: 이상치 제거



---

**참고**: 이 프로젝트는 연구 및 교육 목적으로 개발되었습니다. 실제 운동 지도나 의료 목적으로 사용하기 전에 충분한 검증이 필요합니다.

