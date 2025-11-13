# 실시간 스쿼트 분류 추론 파이프라인 상세 설명

## 개요

이 문서는 `realtime_new_pipeline.py`에서 구현된 실시간 스쿼트 자세 분류 시스템의 전체 동작 과정을 상세히 설명합니다.

## 시스템 아키텍처

### 주요 구성 요소

1. **RealTimeClassifier**: 메인 클래스, 전체 파이프라인 오케스트레이션
2. **RepDetector**: 1-rep 스쿼트 감지 및 상태 머신 (FSM)
3. **SmoothingPipeline**: 확률 스무딩 및 불확실성 처리
4. **CameraWorker**: 카메라 피드 처리 및 MediaPipe 기반 분류
5. **TFLiteBackend**: TensorFlow Lite 모델 추론 백엔드

---

## 1. 데이터 수신 및 버퍼링

### 1.1 UDP 패킷 수신

```python
# ESP32에서 전송되는 센서 데이터 수신
# 패킷 형식: [s0_ax, s0_ay, s0_az, s0_gx, s0_gy, s0_gz,
#             s1_ax, s1_ay, s1_az, s1_gx, s1_gy, s1_gz,
#             s2_ax, s2_ay, s2_az, s2_gx, s2_gy, s2_gz]
```

- **샘플링 레이트**: 110 Hz (기본값)
- **센서 구성**: 3개 (s0: 허리, s1: 허벅지, s2: 종아리)
- **각 센서 데이터**: 가속도 3축 (ax, ay, az) + 자이로 3축 (gx, gy, gz) = 총 18차원

### 1.2 입력 버퍼

```python
self.input_buffer = deque(maxlen=512)  # 고정 길이 원형 버퍼
```

- **크기**: 512 샘플 (≈ 4.65초 @ 110Hz)
- **구조**: `(timestamp, values_array)` 튜플 저장
- **동작**: 버퍼가 가득 차면 자동으로 오래된 샘플 제거 (FIFO)

---

## 2. Rep 감지 (Rep Detection)

### 2.1 센서 기반 Rep 감지

Rep 감지는 **s1_gz** (허벅지 센서의 각속도 Z축)를 기반으로 수행됩니다.

#### 2.1.1 저역통과 필터 (Low-Pass Filter)

```python
def _lowpass_filter_gz(self, gz: float) -> float:
    # 이동평균 필터 (윈도우 크기: 15 샘플)
    filtered = np.mean(list(self.gz_buffer))
```

- **목적**: 노이즈 제거 및 신호 안정화
- **방법**: 15 샘플 이동평균

#### 2.1.2 상태 머신 (FSM)

Rep 감지는 다음 4가지 상태로 구성된 FSM을 사용합니다:

1. **STATE_IDLE**: 대기 상태 (각속도 ≈ 0)
2. **STATE_DESCENT**: 하강 중 (gz > 0)
3. **STATE_BOTTOM**: 바닥 상태 (각속도 ≈ 0, 하강과 상승 사이)
4. **STATE_ASCENT**: 상승 중 (gz < 0)

#### 2.1.3 상태 전이 패턴

```
Idle (0) → Descent (gz > 0) → Bottom (0 부근) → Ascent (gz < 0) → Idle (0, rep 완료)
```

**전이 조건**:
- **Idle → Descent**: `gz > gz_positive_threshold` (기본값: 0.0)
- **Descent → Bottom**: `abs(gz) < gz_zero_threshold` (기본값: 0.1)
- **Bottom → Ascent**: `gz < gz_negative_threshold` (기본값: 0.0)
- **Ascent → Idle**: `abs(gz) < dynamic_zero_threshold` (하강 최댓값의 50%)

### 2.2 s0_gy 기반 Rep 검증

걷기 등 작은 동작을 필터링하기 위해 **s0_gy** (허리 센서의 각속도 Y축) 검증을 수행합니다.

#### 2.2.1 검증 조건

- **타임아웃**: rep 시작 후 1초 이내
- **임계값**: s0_gy가 rep 시작 시점 대비 **0.5 이상 증가**해야 함
- **결과**: 검증 실패 시 rep 취소 (rep_id 증가하지 않음)

#### 2.2.2 검증 로직

```python
def update_s0_gy(self, s0_gy: float, timestamp: float) -> bool:
    # 1. rep 시작 시점의 s0_gy 저장
    if self.s0_gy_at_rep_start is None:
        self.s0_gy_at_rep_start = s0_gy
    
    # 2. 검증 기간 동안 최대 증가량 추적
    s0_gy_increase = s0_gy - self.s0_gy_at_rep_start
    if s0_gy_increase > self.s0_gy_max_in_validation:
        self.s0_gy_max_in_validation = s0_gy_increase
    
    # 3. 0.5 이상 증가 시 검증 통과
    if self.s0_gy_max_in_validation >= 0.5:
        self.rep_validated = True
    
    # 4. 1초 경과 후 검증 실패 시 False 반환
    if elapsed > 1.0 and not self.rep_validated:
        return False
```

### 2.3 Rep ID 관리

- **rep_id 증가 시점**: rep이 **완료될 때만** 증가 (`_finalize_rep`에서)
- **검증 실패 시**: rep_id 증가하지 않음 (rep 취소)
- **결과**: 실제 완료된 rep만 번호가 부여됨

---

## 3. 슬라이딩 윈도우 추론

### 3.1 트리거 조건

```python
if len(self.input_buffer) == self.args.window_size:  # 512 샘플
    if self.global_step - self._last_infer_step >= self.stride_samples:
        # 추론 수행
```

- **윈도우 크기**: 512 샘플 (≈ 4.65초)
- **스트라이드**: 기본값 0.5초 (55 샘플 @ 110Hz)
- **조건**: 버퍼가 가득 차고, 마지막 추론으로부터 stride_samples 이상 경과

### 3.2 전처리 (Preprocessing)

#### 3.2.1 Scaler 변환

```python
# 학습 시 사용한 StandardScaler 적용
window_scaled = self.scaler.transform(window_np)
```

#### 3.2.2 Per-Window Z-score 정규화

```python
# 각 윈도우 내에서 독립적으로 정규화
mean = np.mean(window_scaled, axis=0, keepdims=True)
std = np.std(window_scaled, axis=0, keepdims=True) + 1e-6
window_zscore = (window_scaled - mean) / std
```

#### 3.2.3 ±6σ 클리핑

```python
# 이상치 제거
window_clipped = np.clip(window_zscore, -6.0, 6.0)
```

### 3.3 모델 추론

```python
# TensorFlow Lite 모델 추론
logits = self.backend.predict(window_clipped[None, ...])
probs = softmax(logits[0])  # logits → softmax 확률
```

- **입력**: `[1, 512, 18]` (배치, 시간, 특징)
- **출력**: `[5]` (5개 클래스의 확률)

### 3.4 클래스 정의

```python
CLASS_LABELS_EN = {
    0: "Correct",           # 정상 자세
    1: "Knee Valgus",      # 무릎 안쪽 무너짐
    2: "Forward Lean",      # 앞으로 기울임
    3: "Knee Cave",         # 무릎 안쪽 접힘
    4: "Partial Squat",     # 얕은 스쿼트
}
```

---

## 4. 스무딩 파이프라인 (SmoothingPipeline)

### 4.1 EMA (Exponential Moving Average)

```python
# 확률 스무딩
self.ema_probs = self.ema_alpha * probs + (1 - self.ema_alpha) * self.ema_probs
```

- **α (alpha)**: 기본값 0.3 (낮을수록 더 부드러움)
- **목적**: 급격한 분류 변화 완화

### 4.2 불확실성 보류 (Uncertainty Holding)

```python
p_max = np.max(self.ema_probs)
entropy = -np.sum(self.ema_probs * np.log(self.ema_probs + 1e-10))

is_transition = (p_max < self.uncertainty_p_max) or (entropy > self.uncertainty_h)
```

- **p_max 임계값**: 기본값 0.25 (최대 확률이 낮으면 불확실)
- **엔트로피 임계값**: 기본값 1.6 (엔트로피가 높으면 불확실)
- **결과**: 불확실한 경우 `TRANSITION` 라벨 부여

### 4.3 다수결 투표 (Majority Voting)

```python
# 최근 k개 창의 분류 결과 투표
self.majority_buffer.append(class_id)
if len(self.majority_buffer) >= self.majority_k:
    confirmed_label = Counter(self.majority_buffer).most_common(1)[0][0]
```

- **k 값**: 기본값 3 (최근 3개 창)
- **목적**: 일시적인 오분류 필터링

---

## 5. Rep 레벨 통합 및 리샘플링 추론

### 5.1 Rep 샘플 수집

```python
# rep 진행 중 모든 샘플 저장
if self.rep_detector.rep_start_t is not None:
    self.rep_samples.append((timestamp, values_arr))
```

### 5.2 Rep 완료 시 리샘플링 추론

rep이 완료되면, 해당 rep 구간의 모든 샘플을 **512 샘플로 리샘플링**하여 전체 rep에 대한 최종 분류를 수행합니다.

```python
def _resample_rep_samples(self) -> np.ndarray:
    # 선형 보간을 사용하여 512 샘플로 리샘플링
    timestamps = [ts for ts, _ in self.rep_samples]
    values_list = [vals for _, vals in self.rep_samples]
    
    # 새로운 시간축 생성 (512 샘플)
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], 512)
    
    # 선형 보간
    resampled = np.array([
        np.interp(new_timestamps, timestamps, [v[i] for v in values_list])
        for i in range(18)
    ]).T
    
    return resampled
```

### 5.3 Rep 최종 분류

```python
# 리샘플링된 데이터로 추론
resampled_window = self._resample_rep_samples()
resampled_label, resampled_conf = self._infer_rep_samples()
```

- **목적**: 전체 rep에 대한 더 정확한 분류
- **결과**: rep 완료 시 창-레벨 분류와 리샘플링 분류 모두 출력

### 5.4 Partial Squat 체크

```python
# 실시간 분류 또는 리샘플링 분류 중 하나라도 Partial Squat이면 최종 결과를 Partial Squat으로 표기
final_label = rep_label
if rep_label == 4 or resampled_label == 4:
    final_label = 4  # Partial Squat
```

---

## 6. 카메라 통합

### 6.1 MediaPipe 기반 분류

- **입력**: 카메라 프레임
- **처리**: MediaPipe Pose Estimation
- **분류**: 2-class (Good/Bad)
- **임계값**: 기본값 0.5

### 6.2 실시간 오버레이

#### 6.2.1 Rep이 아닐 때

- 현재 분류 표시하지 않음
- 카메라 분류 결과만 파란색으로 하단 표시

#### 6.2.2 Rep 진행 중

```
Descent: Correct
1rep: ...
Camera: Correct
```

- 현재 상태 (Descent/Bottom/Ascent)와 분류 결과 표시
- rep 번호와 "..." 표시

#### 6.2.3 Rep 완료 후

```
1rep: Partial Squat
Camera: Correct
```

- 완료된 rep 번호와 최종 분류 결과 표시

### 6.3 Rep 클립 저장

```python
# rep 완료 시 해당 구간의 비디오 클립 저장
clip_path = clips_dir / f"{session_id}_rep_{rep_id}.mp4"
self.camera.save_clip(start_wall, end_wall, clip_path)
```

- **저장 위치**: `logs/realtime_sessions/clips/`
- **파일명**: `{session_id}_rep_{rep_id}.mp4`
- **재생**: 'r' 키를 누르면 최근 rep 클립을 0.5배속으로 재생

---

## 7. 특수 처리 로직

### 7.1 Knee Valgus 하강 구간 억제

```python
# Descent 상태에서 Knee Valgus (class 1) 감지 시 Correct로 변경
if self.rep_detector.state == self.rep_detector.STATE_DESCENT and fused_class == 1:
    fused_class = 0  # Correct
    class_id = 0
```

- **이유**: 하강 구간에서는 무릎 안쪽 무너짐이 정상일 수 있음

### 7.2 Partial Squat 표시

- **실시간 분류**: "Keep going"으로 표시
- **리샘플링 분류**: "Partial Squat"으로 표시

---

## 8. 출력 및 로깅

### 8.1 콘솔 출력

#### 8.1.1 창-레벨 출력 (Rep 진행 중만)

```
[Window] t=12.34s, class=0 (Correct), p_max=0.85, H=0.45, gz=0.23, state=Descent
           📍 Rep #1 진행 중
```

#### 8.1.2 Rep 완료 출력

```
🎯 Rep #1 완료: Correct (confidence: 0.82)
   📊 리샘플링 추론: Correct (confidence: 0.88)
   시간: 10.50s~15.20s, 분포: {0: 45, 1: 3, 2: 2}
```

### 8.2 그래프 저장

종료 시 (`Ctrl+C`) 다음 그래프가 저장됩니다:

1. **s1_gz (각속도 Z축)**: rep 상태 및 구간 표시
2. **Energy (s0, s1, s2)**: 각 센서의 가속도 에너지
3. **Confidence**: 창-레벨 분류 신뢰도

**저장 위치**: `logs/realtime_sessions/session_YYYYMMDD_HHMMSS.png`

---

## 9. 전체 파이프라인 흐름도

```
[UDP 패킷 수신]
    ↓
[입력 버퍼링 (512 샘플)]
    ↓
[Rep 감지 (s1_gz 기반 FSM)]
    ↓
[Rep 검증 (s0_gy 기반, 1초 내 0.5 이상 증가 확인)]
    ↓ (검증 실패 시 rep 취소)
[슬라이딩 윈도우 추론 (stride 0.5초)]
    ↓
[전처리: Scaler → Z-score → ±6σ 클리핑]
    ↓
[TFLite 모델 추론 (logits → softmax)]
    ↓
[스무딩: EMA → 불확실성 보류 → 다수결 투표]
    ↓
[Rep 레벨 통합]
    ↓
[Rep 완료 시 리샘플링 추론 (512 샘플)]
    ↓
[카메라 오버레이 및 클립 저장]
    ↓
[콘솔 출력 및 그래프 저장]
```

---

## 10. 주요 파라미터

### 10.1 Rep 감지

- `gz_zero_threshold`: 0.1 (Bottom 구간 판단)
- `gz_positive_threshold`: 0.0 (Descent 판단)
- `gz_negative_threshold`: 0.0 (Ascent 판단)
- `min_rep_duration_sec`: 0.5 (최소 rep 지속 시간)

### 10.2 Rep 검증

- `rep_validation_timeout_sec`: 1.0 (검증 타임아웃)
- `s0_gy_increase_threshold`: 0.5 (검증 임계값)

### 10.3 스무딩

- `ema_alpha`: 0.3 (EMA 스무딩 계수)
- `uncertainty_p_max`: 0.25 (불확실성 p_max 임계값)
- `uncertainty_h`: 1.6 (불확실성 엔트로피 임계값)
- `majority_k`: 3 (다수결 투표 윈도우 크기)

### 10.4 추론

- `window_size`: 512 (윈도우 크기)
- `sliding_stride_sec`: 0.5 (스트라이드)
- `sample_rate_hz`: 110 (샘플링 레이트)

---

## 11. 성능 최적화

### 11.1 지연 최소화

- **스트라이드 조정**: 더 작은 stride (예: 0.25초)로 더 빠른 반응, 하지만 계산량 증가
- **EMA α 조정**: 더 큰 α (예: 0.6)로 더 빠른 반응, 하지만 노이즈 증가

### 11.2 정확도 향상

- **EMA α 감소**: 더 작은 α (예: 0.2)로 더 부드러운 분류
- **majority_k 증가**: 더 큰 k (예: 5)로 더 안정적인 분류

---

## 12. 문제 해결

### 12.1 Rep이 인식되지 않음

- `gz_zero_threshold` 조정 (더 작게)
- `min_rep_duration_sec` 조정 (더 작게)

### 12.2 너무 많은 Rep 인식

- `gz_positive_threshold` 증가 (더 큰 값 필요)
- s0_gy 검증 임계값 증가 (더 큰 값 필요)

### 12.3 분류가 불안정함

- `ema_alpha` 감소
- `majority_k` 증가
- `uncertainty_p_max` 증가 (더 많은 창을 확정으로 처리)

---

## 결론

이 파이프라인은 실시간 센서 데이터를 받아 스쿼트 자세를 분류하고, rep 단위로 통합하여 최종 결과를 제공합니다. s1_gz 기반 rep 감지와 s0_gy 기반 검증을 통해 정확한 rep 인식을 수행하며, 스무딩과 불확실성 처리를 통해 안정적인 분류를 제공합니다.

