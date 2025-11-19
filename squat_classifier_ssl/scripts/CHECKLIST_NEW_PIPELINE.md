# 새로운 실시간 추론 파이프라인 실행 체크리스트

## 실행 전 확인사항

1. **모델 및 스케일러 경로 확인**
   ```bash
   ls -lh squat_classifier_ssl/exports/squat_classifier_fp16.tflite
   ls -lh squat_classifier_ssl/checkpoints/squat_scaler_18axis.pkl
   ```

2. **ESP32 연결 확인**
   - ESP32가 켜져 있고 "ESP32_AP" 네트워크가 보이는지 확인
   - `--skip-wifi-check` 옵션으로 Wi-Fi 체크 건너뛰기 가능

3. **카메라 사용 시**
   - 카메라가 연결되어 있는지 확인
   - `--enable-camera --camera-display` 옵션 사용

## 실행 명령어

```bash
# 기본 실행 (카메라 없이)
python scripts/realtime_new_pipeline.py

# 카메라 포함 실행
python scripts/realtime_new_pipeline.py --enable-camera --camera-display

# 파라미터 조정 예시
python scripts/realtime_new_pipeline.py \
  --sliding-stride-sec 0.25 \
  --ema-alpha 0.6 \
  --uncertainty-p-max 0.50 \
  --uncertainty-h 1.2 \
  --majority-k 5 \
  --fsm-enter 2 \
  --fsm-exit 2 \
  --enable-camera
```

## 실행 시 집중적으로 체크해야 할 부분

### 1. **입력 버퍼링 및 슬라이딩 윈도우 트리거** ⭐⭐⭐
   - **체크 포인트**: 버퍼가 512 샘플에 도달한 후 stride_samples마다 추론이 발생하는지
   - **확인 방법**: 콘솔 출력에서 "🔎 창 #N" 메시지가 일정 간격으로 나타나는지 확인
   - **예상 출력**: 약 0.5초(또는 설정한 stride)마다 창 추론 결과 출력
   - **문제 징후**: 
     - 추론이 너무 자주/드물게 발생
     - 버퍼가 가득 차지 않음

### 2. **전처리 파이프라인** ⭐⭐⭐
   - **체크 포인트**: 
     - scaler.transform() 적용 여부
     - per-window Z-score 정규화
     - ±6σ 클리핑
   - **확인 방법**: 
     - 전처리 후 데이터 범위 확인 (클리핑으로 ±6 범위 내)
     - 학습 시와 동일한 전처리인지 확인
   - **문제 징후**: 
     - 추론 결과가 비정상적
     - NaN 또는 Inf 값 발생

### 3. **EMA 스무딩** ⭐⭐
   - **체크 포인트**: 
     - EMA가 제대로 적용되어 확률이 부드럽게 변화하는지
     - ema_alpha=0.6이 적절한지
   - **확인 방법**: 
     - 연속된 창에서 probs 값이 급격히 변하지 않고 부드럽게 변화하는지
     - p_max 값의 변화 추이 확인
   - **문제 징후**: 
     - 확률이 너무 급격히 변함 (alpha가 너무 크거나 작음)
     - EMA가 초기화되지 않음

### 4. **불확실성 보류 (Uncertainty Hold)** ⭐⭐
   - **체크 포인트**: 
     - p_max < 0.50 또는 H > 1.2일 때 TRANSITION 처리
     - TRANSITION 시 직전 확정 라벨 사용
   - **확인 방법**: 
     - 콘솔 출력에서 p_max와 H 값 확인
     - 불확실한 구간에서도 라벨이 안정적으로 유지되는지
   - **문제 징후**: 
     - 불확실한 구간에서 라벨이 계속 바뀜
     - TRANSITION 처리가 제대로 안 됨

### 5. **다수결 투표 (Majority Vote)** ⭐⭐
   - **체크 포인트**: 
     - k=5 히스토리에서 다수결로 최종 라벨 결정
     - TRANSITION이면 직전 확정 라벨로 채워 투표
   - **확인 방법**: 
     - 연속된 창에서 라벨이 안정적으로 유지되는지
     - 노이즈에 강건한지 확인
   - **문제 징후**: 
     - 라벨이 너무 자주 바뀜
     - 다수결이 제대로 작동하지 않음

### 6. **FSM 상태머신** ⭐⭐⭐
   - **체크 포인트**: 
     - Idle → Descent → Bottom → Ascent → Idle 전이
     - enter=2, exit=2 히스테리시스로 점프 억제
     - rep 시작/종료 감지
   - **확인 방법**: 
     - 콘솔 출력에서 "📍 Rep #N 진행 중" 메시지 확인
     - rep 완료 시 "🎯 Rep #N 완료" 메시지와 분포 확인
     - rep 레코드가 올바르게 생성되는지
   - **문제 징후**: 
     - rep가 너무 자주/드물게 감지됨
     - rep 경계가 부정확함
     - 상태 전이가 너무 빈번함

### 7. **Rep 레벨 통합** ⭐⭐⭐
   - **체크 포인트**: 
     - rep 종료 시 체류 시간 최댓값 클래스를 rep 라벨로 출력
     - rep 레코드에 start_t, end_t, label, confidence, class_distribution 포함
   - **확인 방법**: 
     - rep 완료 메시지에서 분포와 최종 라벨 확인
     - 여러 클래스가 섞여 있을 때 가장 많이 체류한 클래스가 선택되는지
   - **문제 징후**: 
     - rep 라벨이 부정확함
     - 분포가 비정상적

### 8. **카메라 연계** ⭐
   - **체크 포인트**: 
     - 카메라 오버레이에 센서의 5클래스 결과 표시
     - "Keep going" (class 4) 처리
     - rep 클립 자동 저장
   - **확인 방법**: 
     - 카메라 창에서 "Final: [라벨]" 확인
     - rep 완료 시 클립이 저장되는지 확인
   - **문제 징후**: 
     - 오버레이가 업데이트되지 않음
     - 클립이 저장되지 않음

### 9. **출력 및 로깅** ⭐
   - **체크 포인트**: 
     - 창-레벨: timestamp, class_id, p_max, H, probs
     - rep-레벨: rep_id, start_t, end_t, label, confidence, class_distribution
   - **확인 방법**: 
     - 콘솔 출력 형식 확인
     - window_records와 rep_records에 데이터가 쌓이는지
   - **문제 징후**: 
     - 레코드가 누락됨
     - 형식이 잘못됨

### 10. **그래프 저장** ⭐
   - **체크 포인트**: 
     - 종료 시 s0_az 시계열 + per-window confidence 그래프 저장
     - 윈도우별 클래스 색상 표시
   - **확인 방법**: 
     - `logs/realtime_sessions/session_YYYYMMDD_HHMMSS.png` 파일 생성 확인
     - 그래프에서 윈도우 구간과 confidence가 올바르게 표시되는지
   - **문제 징후**: 
     - 그래프가 생성되지 않음
     - 데이터가 누락됨

## 성능 체크 포인트

1. **추론 지연시간**: 창 추론이 실시간으로 처리되는지 (지연 < 100ms)
2. **메모리 사용량**: 버퍼가 메모리를 과도하게 사용하지 않는지
3. **CPU 사용률**: 실시간 처리에 충분한 여유가 있는지

## 문제 발생 시 확인사항

1. **추론이 발생하지 않음**
   - 버퍼가 512 샘플에 도달했는지 확인
   - stride_samples 계산이 올바른지 확인
   - 패킷 수신이 정상인지 확인

2. **라벨이 너무 자주 바뀜**
   - EMA alpha 값 조정 (낮추기)
   - 다수결 k 값 증가
   - FSM enter/exit 임계값 증가

3. **Rep가 감지되지 않음**
   - FSM 상태 전이 로직 확인
   - _class_to_state 매핑이 적절한지 확인
   - enter/exit 임계값 조정

4. **성능 문제**
   - stride_samples 증가 (추론 빈도 감소)
   - 불필요한 로깅 제거
   - 카메라 해상도 낮추기

## 권장 테스트 시나리오

1. **단일 스쿼트 테스트**: 1회 스쿼트로 rep 감지 확인
2. **연속 스쿼트 테스트**: 여러 rep 연속 수행 시 rep 분리 확인
3. **오자세 테스트**: 각 오자세 클래스가 올바르게 분류되는지
4. **불확실 구간 테스트**: 전이 구간에서 안정성 확인
5. **장시간 테스트**: 메모리 누수 및 성능 저하 확인

