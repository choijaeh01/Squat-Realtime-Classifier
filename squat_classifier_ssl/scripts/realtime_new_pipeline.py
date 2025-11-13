#!/usr/bin/env python3
"""
새로운 실시간 스쿼트 분류 파이프라인

기능:
- 슬라이딩 윈도우 기반 실시간 추론 (512 샘플, stride 0.25-0.5초)
- EMA 스무딩 + 불확실성 보류 + 다수결 투표
- FSM 상태머신으로 rep 단위 통합
- 카메라 연계 (선택적)
"""

from __future__ import annotations

import argparse
import socket
import struct
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Optional, Tuple
import os
import select
import logging
import traceback
import threading

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.modeling import build_squat_classifier  # noqa: E402

DEFAULT_TFLITE_MODEL = Path("/home/jae/squat_classifier_ssl/exports/squat_classifier_fp16.tflite")
DEFAULT_KERAS_WEIGHTS = Path("/home/jae/squat_classifier_ssl/checkpoints/squat_classifier_best.weights.h5")

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow 모듈을 찾을 수 없습니다. "
        "`pip install tensorflow` 또는 라즈베리파이용 wheel을 설치한 뒤 다시 시도하세요."
    ) from exc


# -----------------------------------------------------------------------------
# 상수 및 설정
# -----------------------------------------------------------------------------

PACKET_FORMAT = "<4sIIBBB1x18f"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)
MAGIC = b"IMU1"
QUALITY_OK = 0

CLASS_FEEDBACK = {
    0: ("정자세 (Correct)", "✅ 자세가 훌륭합니다! 현재 폼을 유지하세요."),
    1: ("무릎 모임 (Knee Valgus)", "⚠️ 무릎이 안쪽으로 모입니다. 발끝 방향으로 밀어주세요."),
    2: ("벗 윙크 (Butt Wink)", "⚠️ 엉덩이가 말립니다. 허리를 중립으로 유지하세요."),
    3: ("상체 과다 숙임 (Excessive Lean)", "⚠️ 상체가 많이 숙여집니다. 가슴을 열고 시선을 정면에 두세요."),
    4: ("얕은 스쿼트 (Partial Squat)", "⚠️ 깊이가 부족합니다. 더 깊이 앉아보세요."),
}

CLASS_LABELS_EN = {
    0: "Correct",
    1: "Knee Valgus",
    2: "Butt Wink",
    3: "Excessive Lean",
    4: "Partial Squat",
}


# -----------------------------------------------------------------------------
# 유틸리티 함수
# -----------------------------------------------------------------------------

def decode_packet(packet: bytes) -> Tuple[int, int, Tuple[int, int, int], np.ndarray]:
    """ESP32 패킷 디코딩"""
    if len(packet) != PACKET_SIZE:
        raise ValueError(f"패킷 길이가 다릅니다. ({len(packet)} != {PACKET_SIZE})")
    magic, seq, millis, q0, q1, q2, *values = struct.unpack(PACKET_FORMAT, packet)
    if magic != MAGIC:
        raise ValueError("Magic header 불일치")
    return seq, millis, (q0, q1, q2), np.asarray(values, dtype=np.float32)


def get_current_ssid(interface: str = "wlan0") -> Optional[str]:
    """현재 Wi-Fi SSID 확인"""
    result = subprocess.run(["iwconfig", interface], text=True, capture_output=True)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if 'ESSID:"' in line:
            start = line.index('ESSID:"') + len('ESSID:"')
            end = line.find('"', start)
            return line[start:end]
    return None


def ensure_esp32_connection(ssid: str = "ESP32_AP", interface: str = "wlan0") -> None:
    """ESP32 AP 연결 확인 및 연결"""
    current = get_current_ssid(interface)
    if current == ssid:
        print(f"[Wi-Fi] 이미 {ssid}에 연결되어 있습니다.")
        return

    print(f"[Wi-Fi] 현재 SSID: {current or '확인 불가'}, {ssid}로 전환합니다...")
    subprocess.run(["sudo", "nmcli", "device", "wifi", "rescan"], check=False)
    time.sleep(2.0)
    wifi_list = subprocess.run(["sudo", "nmcli", "device", "wifi", "list"], text=True, capture_output=True)
    if ssid not in wifi_list.stdout:
        raise RuntimeError(f"{ssid} AP를 찾을 수 없습니다.")
    connect = subprocess.run(["sudo", "nmcli", "connection", "up", ssid], text=True, capture_output=True)
    if connect.returncode != 0:
        raise RuntimeError(f"{ssid} 연결 실패: {connect.stderr.strip()}")
    print(f"[Wi-Fi] {ssid} 연결 성공.")


# -----------------------------------------------------------------------------
# 추론 백엔드
# -----------------------------------------------------------------------------

@dataclass
class InferenceBackend:
    window_size: int
    feature_dim: int
    num_classes: int
    use_tflite: bool
    model_path: Path
    dropout: float = 0.3

    def __post_init__(self) -> None:
        if self.use_tflite:
            self._init_tflite()
        else:
            self._init_keras()

    def _init_keras(self) -> None:
        self.model = build_squat_classifier(
            input_shape=(self.window_size, self.feature_dim),
            num_classes=self.num_classes,
            dropout=self.dropout,
        )
        self.model.build((None, self.window_size, self.feature_dim))
        self.model.load_weights(self.model_path)

    def _init_tflite(self) -> None:
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """배치 추론 (logits 반환)"""
        if self.use_tflite:
            self.interpreter.set_tensor(self.input_index, batch)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_index)
        return self.model(batch, training=False).numpy()


# -----------------------------------------------------------------------------
# 스무딩 파이프라인
# -----------------------------------------------------------------------------

class SmoothingPipeline:
    """EMA + 불확실성 보류 + 다수결 스무딩"""

    def __init__(
        self,
        ema_alpha: float = 0.3,  # 스무딩 감소 (0.6 → 0.3)
        uncertainty_p_max: float = 0.50,
        uncertainty_h: float = 1.2,
        majority_k: int = 3,  # 다수결 감소 (5 → 3)
    ):
        self.ema_alpha = ema_alpha
        self.uncertainty_p_max = uncertainty_p_max
        self.uncertainty_h = uncertainty_h
        self.majority_k = majority_k
        self.ema_probs: Optional[np.ndarray] = None
        self.majority_history: Deque[int] = deque(maxlen=majority_k)
        self.last_confirmed_label: Optional[int] = None

    def update(self, probs: np.ndarray) -> Tuple[int, bool]:
        """
        probs: [5] softmax 확률
        Returns: (확정 라벨, is_transition 여부)
        """
        # 1. EMA
        if self.ema_probs is None:
            self.ema_probs = probs.copy()
        else:
            self.ema_probs = self.ema_alpha * probs + (1 - self.ema_alpha) * self.ema_probs

        # 2. 불확실성 판정
        p_max = float(np.max(self.ema_probs))
        h = float(-np.sum(self.ema_probs * np.log(self.ema_probs + 1e-10)))
        is_transition = p_max < self.uncertainty_p_max or h > self.uncertainty_h

        if is_transition:
            # TRANSITION: argmax 사용 (직전 확정 라벨 사용하지 않음)
            # 직전 확정 라벨을 사용하면 초기 잘못된 분류가 계속 유지됨
            confirmed = int(np.argmax(self.ema_probs))
            # TRANSITION이지만 argmax는 majority에 포함 (라벨 업데이트는 안 함)
            self.majority_history.append(confirmed)
            return confirmed, True
        else:
            # 확정: argmax 사용
            confirmed = int(np.argmax(self.ema_probs))
            self.last_confirmed_label = confirmed
            self.majority_history.append(confirmed)

        # 3. 다수결
        if len(self.majority_history) > 0:
            counts = Counter(self.majority_history)
            result = counts.most_common(1)[0][0]
            return result, is_transition
        return int(np.argmax(self.ema_probs)), is_transition


# -----------------------------------------------------------------------------
# FSM 상태머신
# -----------------------------------------------------------------------------

class RepDetector:
    """
    s1_gz(각속도 Z축) 기반 rep 감지기
    s1_gz 값을 저역통과 필터로 필터링하고, 사인함수 패턴(원점→골→원점→마루→원점)으로 1 rep를 인식
    패턴: 0 (원점) → 음수(골, 하강) → 0 (원점) → 양수(마루, 상승) → 0 (원점, rep 완료)
    """

    STATE_IDLE = "Idle"  # 속도가 0 부근 (서있는 자세)
    STATE_DESCENT = "Descent"  # 속도가 음수 (하강 중)
    STATE_BOTTOM = "Bottom"  # 속도가 0 부근 (앉은 자세, 하강 후)
    STATE_ASCENT = "Ascent"  # 속도가 양수 (상승 중)

    def __init__(
        self,
        sample_rate_hz: float = 110.0,
        vel_zero_threshold: float = 0.3,
        vel_negative_threshold: float = -0.3,  # 더 민감하게 (하강 감지 개선)
        vel_positive_threshold: float = 0.3,  # 더 민감하게 (상승 감지 개선)
        min_rep_duration_sec: float = 0.5,  # 최소 지속 시간 감소 (더 빠른 rep 감지)
    ):
        """
        sample_rate_hz: 샘플링 주파수
        vel_zero_threshold: 속도가 0 부근으로 간주하는 임계값
        vel_negative_threshold: 하강 구간 판단 임계값
        vel_positive_threshold: 상승 구간 판단 임계값
        min_rep_duration_sec: rep로 인정하기 위한 최소 지속 시간
        """
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz
        self.vel_zero_threshold = vel_zero_threshold
        self.vel_negative_threshold = vel_negative_threshold
        self.vel_positive_threshold = vel_positive_threshold
        self.min_rep_duration_sec = min_rep_duration_sec

        # s1_gz 기반 rep 감지 (저역통과 필터 적용)
        self.lpf_window = 15  # 저역통과 필터 윈도우 크기
        self.gz_buffer: Deque[float] = deque(maxlen=self.lpf_window)  # 각속도 Z축 버퍼
        self.filtered_gz = 0.0  # 필터링된 각속도 Z축 값
        self.last_gz = 0.0  # 이전 각속도 Z축 값 (FSM 판단용)
        
        # 드리프트 방지 (0 리셋 방식)
        self.idle_reset_threshold = 0.2  # 속도 리셋 임계값
        self.idle_acc_threshold = 0.5  # 가속도 리셋 임계값
        self.velocity_drift_counter = 0  # 드리프트 카운터
        self.idle_stable_samples = 0  # Idle 상태 안정 샘플 수
        self.min_idle_stable_samples = int(sample_rate_hz * 0.5)  # 리셋 전 최소 안정 샘플 수 (0.5초)
        
        # s1_gz 패턴 기반 rep 감지
        # gz > 0: Descent (하강), gz < 0: Ascent (상승)
        self.gz_zero_threshold = 0.1  # 0 부근 판단 각속도 임계값 (rad/s, Bottom 구간)
        self.gz_positive_threshold = 0.0  # Descent 판단 각속도 임계값 (양수)
        self.gz_negative_threshold = 0.0  # Ascent 판단 각속도 임계값 (음수)

        # 상태 머신
        self.state = self.STATE_IDLE
        self.rep_id = 0
        self.rep_start_t: Optional[float] = None
        self.rep_class_counts: dict[int, int] = {}
        self.rep_records: list[dict] = []

        # 패턴 감지 플래그
        self.has_descent = False  # 하강 구간 경험
        self.has_zero_cross = False  # 0 통과 경험
        self.has_peak = False  # 양수 피크 경험
        self.max_descent_gz = 0.0  # 하강 구간에서 각속도 절대값 최댓값
        self.bottom_start_t: Optional[float] = None  # Bottom 상태 시작 시간
        self.bottom_timeout_sec = 3.0  # Bottom 상태 최대 유지 시간 (초과 시 리셋)
        
        # s0_gy 검증 (rep 시작 후 1초 안에 s0_gy가 0.5 이상 증가하지 않으면 rep 취소)
        self.rep_validation_timeout_sec = 1.0  # rep 검증 타임아웃 (1초)
        self.s0_gy_at_rep_start: Optional[float] = None  # rep 시작 시점의 s0_gy 값
        self.s0_gy_max_in_validation: float = 0.0  # 검증 기간 동안의 s0_gy 최댓값
        self.rep_validated = False  # rep이 검증되었는지 여부

    def _lowpass_filter_gz(self, gz: float) -> float:
        """
        s1_gz 값에 저역통과 필터 적용 (이동평균)
        """
        self.gz_buffer.append(gz)
        if len(self.gz_buffer) < self.lpf_window:
            # 버퍼가 부족하면 원본 값 반환
            return gz
        # 이동평균 (저역통과 필터)
        filtered = np.mean(list(self.gz_buffer))
        return filtered

    def update_gz(self, gz: float, timestamp: float) -> float:
        """
        매 샘플마다 s1_gz 업데이트 (저역통과 필터 적용)
        gz: s1 센서의 각속도 Z축 값
        Returns: 필터링된 각속도 Z축 값
        """
        # 저역통과 필터 적용
        self.filtered_gz = self._lowpass_filter_gz(gz)
        self.last_gz = self.filtered_gz
        
        return self.filtered_gz
    
    def update_s0_gy(self, s0_gy: float, timestamp: float) -> bool:
        """
        매 샘플마다 s0_gy 업데이트 및 rep 검증
        s0_gy: s0 센서의 각속도 Y축 값
        timestamp: 현재 타임스탬프
        Returns: rep이 유효한지 여부 (False면 rep 취소 필요)
        """
        # rep이 시작되지 않았거나 이미 검증된 경우
        if self.rep_start_t is None or self.rep_validated:
            return True
        
        # rep 시작 후 1초 이내인지 확인
        elapsed = timestamp - self.rep_start_t
        if elapsed > self.rep_validation_timeout_sec:
            # 1초가 지났는데도 검증되지 않았으면 rep 취소
            if not self.rep_validated:
                return False
            return True
        
        # rep 시작 시점의 s0_gy 값 저장 (첫 번째 값)
        if self.s0_gy_at_rep_start is None:
            self.s0_gy_at_rep_start = s0_gy
        
        # 검증 기간 동안 s0_gy 최댓값 추적
        s0_gy_increase = s0_gy - self.s0_gy_at_rep_start
        if s0_gy_increase > self.s0_gy_max_in_validation:
            self.s0_gy_max_in_validation = s0_gy_increase
        
        # s0_gy가 0.5 이상 증가했으면 rep 검증 완료
        if self.s0_gy_max_in_validation >= 0.5:
            self.rep_validated = True
        
        return True

    def update(self, class_id: int, timestamp: float, is_transition: bool = False) -> Optional[dict]:
        """
        창 단위로 rep 패턴 감지
        Returns: rep 종료 시 rep 레코드, 아니면 None
        """
        # TRANSITION 창도 rep에 포함 (rep 감지 정확도 향상)
        # 상태 전이 처리
        rep_record = self._update_state(class_id, timestamp)
        return rep_record

    def _update_state(self, class_id: int, timestamp: float) -> Optional[dict]:
        """
        s1_gz 패턴 기반 상태 전이
        gz > 0: Descent (하강)
        gz < 0: Ascent (상승)
        Bottom: 0 부근 (gz_zero_threshold 내) - Descent와 Ascent 사이에 겹치도록 설정
        패턴: Idle (0) → Descent (gz > 0) → Bottom (0 부근) → Ascent (gz < 0) → Idle (0, rep 완료)
        """
        gz = self.filtered_gz

        if self.state == self.STATE_IDLE:
            # Idle: 각속도가 0 부근 (원점, 서있는 자세)
            if abs(gz) < self.gz_zero_threshold:
                # 계속 Idle
                pass
            elif gz > self.gz_positive_threshold:
                # 각속도가 양수 → Descent (하강) 시작
                self.state = self.STATE_DESCENT
                # rep_id는 검증 통과 후에만 증가 (검증 실패 시 취소되므로)
                # 임시로 rep_id를 증가시키지 않고, 검증 통과 후 또는 rep 완료 시 증가
                self.rep_start_t = timestamp
                self.rep_class_counts = {class_id: 1}
                self.has_descent = True
                self.has_zero_cross = False
                self.has_peak = False
                self.max_descent_gz = abs(gz)  # 하강 시작 시 초기 최댓값 설정
                # s0_gy 검증 초기화
                self.s0_gy_at_rep_start = None  # 아직 s0_gy 값을 받지 않음
                self.s0_gy_max_in_validation = 0.0
                self.rep_validated = False
            elif gz < self.gz_negative_threshold:
                # 각속도가 음수로 바로 가는 경우는 무시 (노이즈, 정상적인 패턴이 아님)
                pass

        elif self.state == self.STATE_DESCENT:
            # Descent: 각속도가 양수 (하강 중)
            self.rep_class_counts[class_id] = self.rep_class_counts.get(class_id, 0) + 1
            
            # 하강 구간에서 각속도 최댓값 추적
            if gz > self.max_descent_gz:
                self.max_descent_gz = gz

            if gz > self.gz_positive_threshold:
                # 계속 하강 (양수)
                pass
            elif abs(gz) < self.gz_zero_threshold:
                # 각속도가 0 부근으로 복귀 → Bottom (원점, 앉은 자세)
                self.state = self.STATE_BOTTOM
                self.has_zero_cross = True
                self.bottom_start_t = timestamp  # Bottom 상태 시작 시간 기록
            elif gz < self.gz_negative_threshold:
                # 각속도가 음수로 전환 → Ascent (상승) 시작 (Bottom을 거치지 않고 바로 상승)
                self.state = self.STATE_ASCENT
                self.has_zero_cross = True

        elif self.state == self.STATE_BOTTOM:
            # Bottom: 각속도가 0 부근 (원점, 앉은 자세, 하강 후)
            # Bottom은 Descent와 Ascent 사이에 겹치도록 설정 (명확하지 않을 때)
            self.rep_class_counts[class_id] = self.rep_class_counts.get(class_id, 0) + 1
            
            # Bottom 상태에서 너무 오래 유지되면 리셋 (rep 중이지만 비정상 상황)
            if self.bottom_start_t is not None:
                bottom_duration = timestamp - self.bottom_start_t
                if bottom_duration > self.bottom_timeout_sec:
                    # 시간 초과: 리셋 (비정상 상황 처리)
                    self.bottom_start_t = timestamp  # 리셋 시간 갱신

            if abs(gz) < self.gz_zero_threshold:
                # 계속 Bottom (0 부근)
                pass
            elif gz < self.gz_negative_threshold:
                # 각속도가 음수로 전환 → Ascent (상승) 시작
                self.state = self.STATE_ASCENT
                self.bottom_start_t = None
            elif gz > self.gz_positive_threshold:
                # 각속도가 양수로 다시 증가 (다시 하강, 비정상일 수 있음)
                # 하지만 일부 경우 정상일 수 있으므로 Descent로 전환
                self.state = self.STATE_DESCENT
                self.bottom_start_t = None

        elif self.state == self.STATE_ASCENT:
            # Ascent: 각속도가 음수 (상승 중)
            self.rep_class_counts[class_id] = self.rep_class_counts.get(class_id, 0) + 1

            if gz < self.gz_negative_threshold:
                # 계속 상승 (음수, 피크 추적)
                if not self.has_peak or abs(gz) > abs(self.last_gz):
                    self.has_peak = True
            elif abs(gz) < self.gz_zero_threshold:
                # 각속도가 0 부근으로 복귀 → Idle (원점, 서있는 자세, rep 완료)
                if self.has_descent and self.has_zero_cross and self.rep_start_t is not None:
                    duration = timestamp - self.rep_start_t
                    if duration >= self.min_rep_duration_sec:
                        rep_record = self._finalize_rep(timestamp)
                        # rep 완료 시 리셋 (Idle 상태로 복귀)
                        self._reset_state()
                        return rep_record
                # 조건 미충족 시 리셋
                self._reset_state()
            elif gz > self.gz_positive_threshold:
                # 각속도가 양수로 전환 (다시 하강, 비정상, 리셋)
                self._reset_state()

        return None

    def _reset_state(self) -> None:
        """상태 리셋"""
        self.state = self.STATE_IDLE
        self.rep_start_t = None
        self.rep_class_counts = {}
        self.has_descent = False
        self.has_zero_cross = False
        self.has_peak = False
        self.max_descent_gz = 0.0  # 하강 각속도 최댓값 리셋
        self.bottom_start_t = None  # Bottom 상태 시작 시간 리셋
        # s0_gy 검증 관련 변수 리셋
        self.s0_gy_at_rep_start = None
        self.s0_gy_max_in_validation = 0.0
        self.rep_validated = False

    def _finalize_rep(self, end_t: float) -> dict:
        """rep 종료 및 레코드 생성"""
        # rep 완료 시에만 rep_id 증가 (검증 실패로 취소된 rep은 rep_id가 증가하지 않음)
        self.rep_id += 1
        rep_label = max(self.rep_class_counts.items(), key=lambda x: x[1])[0]
        rep_confidence = max(self.rep_class_counts.values()) / sum(self.rep_class_counts.values())
        rep_record = {
            "rep_id": self.rep_id,
            "start_t": self.rep_start_t,
            "end_t": end_t,
            "label": rep_label,
            "confidence": rep_confidence,
            "class_distribution": dict(self.rep_class_counts),
        }
        self.rep_records.append(rep_record)
        return rep_record

    def finalize_current(self) -> Optional[dict]:
        """현재 진행 중인 rep를 강제 종료"""
        if self.rep_start_t is not None and self.state != self.STATE_IDLE:
            return self._finalize_rep(time.time())
        return None

    def reset_velocity(self) -> None:
        """속도 누적값 리셋 (드리프트 방지)"""
        # Idle 상태일 때만 리셋
        if self.state == self.STATE_IDLE and abs(self.velocity) < self.vel_zero_threshold:
            self.velocity = 0.0
            self.last_update_t = None


# -----------------------------------------------------------------------------
# 카메라 워커
# -----------------------------------------------------------------------------

class CameraWorker:
    """카메라 기반 Good/Bad 분류 및 오버레이"""

    def __init__(
        self,
        model_path: Path,
        classification_threshold: float = 0.7,
        cam_width: int = 640,
        cam_height: int = 360,
        buffer_seconds: float = 12.0,
        use_picamera2: bool = True,
        rotate_deg: int = 0,
        draw_landmarks: bool = True,
    ):
        self.model_path = Path(model_path)
        self.threshold = float(classification_threshold)
        self.cam_width = int(cam_width)
        self.cam_height = int(cam_height)
        self.buffer_seconds = float(buffer_seconds)
        self.use_picamera2 = bool(use_picamera2)
        self.display_enabled = False
        self.rotate_deg = int(rotate_deg) % 360
        self.draw_landmarks = bool(draw_landmarks)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._overlay_text = "Initializing..."
        self._overlay_color = (0, 0, 255)
        self._person_present = False
        self._last_is_bad = False
        self._last_cam_label = "NoPerson"
        self._last_cam_score = 0.0
        self._frame_buf: Deque[tuple[float, np.ndarray]] = deque(maxlen=int(self.buffer_seconds * 30))
        self.start_event = threading.Event()
        self._last_clip_path: Optional[Path] = None
        self._session_id: Optional[str] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def set_overlay(self, text: str, color: tuple[int, int, int]) -> None:
        with self._lock:
            self._overlay_text = text
            self._overlay_color = tuple(int(c) for c in color)

    def get_camera_state(self) -> tuple[bool, bool, str]:
        """returns (person_present, is_bad, cam_label_en)"""
        with self._lock:
            return self._person_present, self._last_is_bad, self._last_cam_label

    def save_clip(self, start_ts: float, end_ts: float, out_path: Path) -> Optional[Path]:
        """지정된 시간 구간의 클립 저장"""
        if start_ts >= end_ts:
            return None
        frames: list[np.ndarray] = []
        with self._lock:
            for ts, fr in list(self._frame_buf):
                if start_ts <= ts <= end_ts:
                    frames.append(fr.copy())
        if not frames:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import cv2
        except Exception:
            return None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]))
        if not writer.isOpened():
            return None
        for fr in frames:
            writer.write(fr)
        writer.release()
        return out_path

    def _run_loop(self) -> None:
        try:
            import cv2
            import mediapipe as mp
            import tensorflow as tf
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            mp_drawing = mp.solutions.drawing_utils
        except Exception:
            return

        interpreter = None
        input_details = None
        output_details = None
        input_dtype = None
        max_seq_len = 218
        try:
            interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_dtype = input_details[0]["dtype"]
            max_seq_len = int(input_details[0]["shape"][1])
        except Exception:
            interpreter = None

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

        picam2 = None
        cap = None
        use_picam2 = False
        if self.use_picamera2:
            try:
                from picamera2 import Picamera2
                # PipeWire가 카메라를 점유하고 있을 수 있으므로, 카메라 모드 설정
                picam2 = Picamera2()
                # 더 낮은 해상도로 시도 (버퍼 문제 완화)
                config = picam2.create_preview_configuration(
                    main={"size": (self.cam_width, self.cam_height)},
                    controls={"FrameRate": 30}
                )
                picam2.configure(config)
                picam2.start()
                # 초기 프레임 몇 개 버리기 (버퍼 안정화)
                # 버퍼 큐잉 에러는 무시하고 계속 진행
                for _ in range(10):
                    try:
                        _ = picam2.capture_array("main")
                        break  # 성공하면 중단
                    except Exception:
                        time.sleep(0.1)  # 에러 시 잠시 대기
                        continue
                use_picam2 = True
                print("[Camera] Picamera2 initialized successfully")
                print("[Camera] Note: Buffer queue errors may appear but camera should work")
            except Exception as e:
                picam2 = None
                use_picam2 = False
                print(f"[Camera] Picamera2 initialization failed: {e}")
                print(f"[Camera] Error type: {type(e).__name__}")
                import traceback
                print(f"[Camera] Traceback:\n{traceback.format_exc()}")
                print("[Camera] Falling back to USB camera (cv2.VideoCapture)")
        if not use_picam2:
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    # 다른 인덱스 시도
                    for idx in [1, 2, 3]:
                        cap = cv2.VideoCapture(idx)
                        if cap.isOpened():
                            break
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
                    print(f"[Camera] USB camera opened successfully")
                else:
                    print("[Camera] Failed to open any camera device")
            except Exception as e:
                print(f"[Camera] USB camera initialization failed: {e}")
                cap = None

        STATE_WAITING = 0
        STATE_IN_MOTION = 1
        state = STATE_WAITING
        motion_buffer: list[list[float]] = []
        KNEE_DOWN = 150.0
        KNEE_UP = 155.0

        try:
            while not self._stop.is_set():
                frame_bgr = None
                if use_picam2:
                    try:
                        if picam2 is not None:
                            # 버퍼 큐잉 에러가 발생해도 프레임은 캡처 가능
                            frame_bgr = picam2.capture_array("main")
                            if frame_bgr is not None and frame_bgr.ndim == 3:
                                if frame_bgr.shape[2] == 4:
                                    frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)
                                elif frame_bgr.shape[2] == 3:
                                    # 이미 BGR 형식
                                    pass
                    except Exception as e:
                        # 버퍼 큐잉 에러는 무시하고 계속 시도
                        # 에러 메시지는 출력하지 않음 (너무 많이 출력됨)
                        time.sleep(0.05)
                        continue
                else:
                    if cap is None or not cap.isOpened():
                        time.sleep(0.1)
                        continue
                    ok, frame_bgr = cap.read()
                    if not ok or frame_bgr is None:
                        time.sleep(0.01)
                        continue
                
                if frame_bgr is None:
                    time.sleep(0.005)
                    continue

                ts = time.time()
                if self.rotate_deg == 90:
                    frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotate_deg == 180:
                    frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
                elif self.rotate_deg == 270:
                    frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)

                avg_knee_angle = -1.0
                detected = bool(results.pose_landmarks)
                if detected:
                    try:
                        lm = results.pose_landmarks.landmark
                        l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        l_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        r_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        l_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        r_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                        l_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        r_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        l_heel = [lm[mp_pose.PoseLandmark.LEFT_HEEL.value].x, lm[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                        r_heel = [lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]

                        def ang(a, b, c) -> float:
                            a = np.asarray(a)
                            b = np.asarray(b)
                            c = np.asarray(c)
                            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                            deg = abs(radians * 180.0 / np.pi)
                            return 360.0 - deg if deg > 180.0 else deg

                        angle_l_knee = ang(l_hip, l_knee, l_ankle)
                        angle_r_knee = ang(r_hip, r_knee, r_ankle)
                        avg_knee_angle = (angle_l_knee + angle_r_knee) / 2.0
                        angle_l_hip = ang(l_shoulder, l_hip, l_knee)
                        angle_r_hip = ang(r_shoulder, r_hip, r_knee)
                        mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
                        mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
                        vertical_point = [mid_hip[0], mid_hip[1] - 100]
                        torso_angle = ang(mid_shoulder, mid_hip, vertical_point)
                        angle_l_ankle = ang(l_knee, l_ankle, l_heel)
                        angle_r_ankle = ang(r_knee, r_ankle, r_heel)
                        avg_ankle_angle = (angle_l_ankle + angle_r_ankle) / 2.0

                        features = [
                            angle_l_knee, angle_r_knee,
                            angle_l_hip, angle_r_hip,
                            torso_angle,
                            avg_ankle_angle,
                        ]

                        if state == STATE_WAITING:
                            if avg_knee_angle > 0 and avg_knee_angle < KNEE_DOWN:
                                state = STATE_IN_MOTION
                                motion_buffer = []
                        elif state == STATE_IN_MOTION:
                            motion_buffer.append(features)
                            if avg_knee_angle > KNEE_UP:
                                state = STATE_WAITING
                                if interpreter is not None and len(motion_buffer) > 10:
                                    inp = pad_sequences([motion_buffer], maxlen=max_seq_len, dtype="float32", padding="post", truncating="post")
                                    if input_dtype == np.float16:
                                        inp = inp.astype(np.float16)
                                    interpreter.set_tensor(input_details[0]["index"], inp)
                                    interpreter.invoke()
                                    pred = interpreter.get_tensor(output_details[0]["index"])
                                    score = float(pred[0][0])
                                    with self._lock:
                                        self._last_is_bad = not (score > self.threshold)
                                        self._last_cam_score = score
                                        self._last_cam_label = "Correct" if score > self.threshold else "Bad"
                                motion_buffer = []
                    except Exception:
                        pass

                if self.display_enabled and self.draw_landmarks and detected:
                    try:
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            results.pose_landmarks,
                            mp.solutions.pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                        )
                    except Exception:
                        pass

                with self._lock:
                    self._person_present = detected
                    if not detected:
                        self._last_cam_label = "NoPerson"
                    text = self._overlay_text
                    color = self._overlay_color
                    self._frame_buf.append((ts, frame_bgr.copy()))

                try:
                    # DISPLAY 환경 변수 확인 및 설정
                    display_available = bool(os.environ.get("DISPLAY"))
                    if not display_available:
                        # SSH 세션에서 X11 포워딩 시도
                        try:
                            # 여러 DISPLAY 값 시도
                            for disp_val in [":0", ":10.0", "localhost:10.0"]:
                                try:
                                    os.environ["DISPLAY"] = disp_val
                                    # 테스트로 간단한 창 열기 시도
                                    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                                    cv2.imshow("test", test_img)
                                    cv2.waitKey(1)
                                    cv2.destroyAllWindows()
                                    display_available = True
                                    print(f"[Camera] DISPLAY set to {disp_val}")
                                    break
                                except Exception:
                                    continue
                        except Exception as e:
                            if self.display_enabled:
                                print(f"[Camera] DISPLAY setup failed: {e}")
                    
                    if self.display_enabled and display_available:
                        lines = str(text).splitlines()
                        if lines:
                            # 메인 텍스트 (상단)
                            y = 40
                            for i, line in enumerate(lines):
                                if not line:
                                    y += 30
                                    continue
                                # 마지막 줄이 "Camera:"로 시작하면 파란색으로 하단에 표시
                                if i == len(lines) - 1 and line.startswith("Camera:"):
                                    cv2.putText(frame_bgr, line, (30, frame_bgr.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3, cv2.LINE_AA)
                                else:
                                    cv2.putText(frame_bgr, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
                                    y += 34
                        cv2.imshow("Squat Camera (q to quit)", frame_bgr)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                        if key == ord("s"):
                            self.start_event.set()
                        if key == ord("r") and self._last_clip_path:
                            cap2 = cv2.VideoCapture(str(self._last_clip_path))
                            if cap2.isOpened():
                                # 0.5배속 재생 (원래 fps의 절반 속도)
                                fps = cap2.get(cv2.CAP_PROP_FPS)
                                frame_delay = int(1000 / (fps * 0.5))  # 0.5배속
                                while True:
                                    ok2, fr2 = cap2.read()
                                    if not ok2:
                                        break
                                    cv2.imshow("Replay (0.5x speed, q=quit/esc)", fr2)
                                    k2 = cv2.waitKey(frame_delay) & 0xFF
                                    if k2 in (27, ord("q"), ord(" ")):
                                        break
                                cap2.release()
                    elif self.display_enabled and not display_available:
                        # DISPLAY가 없어도 카메라는 작동하도록 (헤드리스 모드)
                        pass
                except Exception as e:
                    # 에러가 발생해도 카메라 스레드는 계속 실행
                    if self.display_enabled:
                        print(f"[Camera] Display error (continuing): {e}")
                    pass

                with self._lock:
                    maxlen = max(1, int(self.buffer_seconds * 30))
                    if self._frame_buf.maxlen != maxlen:
                        self._frame_buf = deque(self._frame_buf, maxlen=maxlen)

        finally:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
            try:
                if picam2 is not None:
                    picam2.stop()
            except Exception:
                pass
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# 실시간 분류기
# -----------------------------------------------------------------------------

class RealTimeClassifier:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.scaler = joblib.load(args.scaler)
        self.backend = InferenceBackend(
            window_size=args.window_size,
            feature_dim=18,
            num_classes=5,
            use_tflite=args.use_tflite,
            model_path=args.model,
            dropout=args.dropout,
        )

        # 입력 버퍼링 (타임스탬프 포함)
        self.input_buffer: Deque[tuple[float, np.ndarray]] = deque(maxlen=args.window_size)
        self.rep_samples: list[tuple[float, np.ndarray]] = []  # rep 구간 샘플 저장 (리샘플링 추론용)
        # 슬라이딩 윈도우 설정
        stride_sec = args.sliding_stride_sec
        self.stride_samples = max(1, int(stride_sec * args.sample_rate_hz))
        # 스무딩 파이프라인
        self.smoothing = SmoothingPipeline(
            args.ema_alpha,
            args.uncertainty_p_max,
            args.uncertainty_h,
            args.majority_k,
        )
        # Rep 감지기 (속도 패턴 기반)
        vel_zero_threshold = getattr(args, "vel_zero_threshold", 0.3)
        vel_negative_threshold = getattr(args, "vel_negative_threshold", -0.5)
        vel_positive_threshold = getattr(args, "vel_positive_threshold", 0.5)
        min_rep_duration_sec = getattr(args, "min_rep_duration_sec", 0.8)
        self.rep_detector = RepDetector(
            sample_rate_hz=args.sample_rate_hz,
            vel_zero_threshold=vel_zero_threshold,
            vel_negative_threshold=vel_negative_threshold,
            vel_positive_threshold=vel_positive_threshold,
            min_rep_duration_sec=min_rep_duration_sec,
        )
        # 클리핑 설정
        self.clip_sigma = args.clip_sigma

        self.global_step = 0
        self.total_windows = 0
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.window_records: list[dict] = []
        self.clip_paths: list[Path] = []
        self._last_logged_rep_id = 0  # 마지막으로 출력한 rep_id 추적
        self._last_rep_info: Optional[dict] = None  # 마지막 완료된 rep 정보 (카메라 오버레이용)
        self._current_rep_id: Optional[int] = None  # 현재 진행 중인 rep_id (카메라 오버레이용)
        # 그래프 저장용 데이터
        self.sample_times: list[float] = []
        self.series_s1_gz: list[float] = []  # s1 각속도 Z축 (rep 감지용)
        self.series_energy_s0: list[float] = []  # s0 에너지 (ax, ay, az 기반)
        self.series_energy_s1: list[float] = []  # s1 에너지 (ax, ay, az 기반)
        self.series_energy_s2: list[float] = []  # s2 에너지 (ax, ay, az 기반)
        self.rep_states: list[Optional[str]] = []  # rep 상태 기록 (그래프용)

        # 카메라 초기화
        self.camera: Optional[CameraWorker] = None
        if args.enable_camera:
            try:
                self.camera = CameraWorker(
                    model_path=args.camera_model,
                    classification_threshold=args.camera_threshold,
                    cam_width=args.camera_width,
                    cam_height=args.camera_height,
                    buffer_seconds=args.clip_buffer_sec,
                    use_picamera2=args.camera_use_picamera2,
                    rotate_deg=args.camera_rotate,
                    draw_landmarks=args.camera_draw_landmarks,
                )
                self.camera.display_enabled = bool(args.camera_display)
                self.camera._session_id = self.session_id
                self.camera.start()
            except Exception as exc:
                print(f"[Camera] 카메라 초기화 실패: {exc}")

    def _prepare_window(self, window: np.ndarray) -> np.ndarray:
        """전처리: scaler → per-window Z-score → ±6σ 클리핑"""
        # 1. scaler.transform()
        scaled = self.scaler.transform(window)
        # 2. per-window Z-score
        if self.args.per_window_zscore:
            mean = scaled.mean(axis=0, keepdims=True)
            std = scaled.std(axis=0, keepdims=True) + 1e-6
            scaled = (scaled - mean) / std
            # 3. ±6σ 클리핑
            scaled = np.clip(scaled, -self.clip_sigma, self.clip_sigma)
        # 길이 조정
        if scaled.shape[0] < self.args.window_size:
            pad = self.args.window_size - scaled.shape[0]
            scaled = np.pad(scaled, ((0, pad), (0, 0)), mode="constant")
        elif scaled.shape[0] > self.args.window_size:
            scaled = scaled[-self.args.window_size :]
        return np.expand_dims(scaled.astype(np.float32, copy=False), axis=0)

    def _resample_rep_samples(self) -> Optional[np.ndarray]:
        """
        rep 구간 샘플을 512 샘플로 리샘플링
        Returns: [512, 18] 배열 또는 None
        """
        if len(self.rep_samples) < 2:
            return None
        
        # 원본 샘플 추출
        original_samples = np.array([v for _, v in self.rep_samples], dtype=np.float32)  # [N, 18]
        original_length = len(original_samples)
        target_length = self.args.window_size  # 512
        
        if original_length == target_length:
            return original_samples
        
        # 리샘플링: 각 축에 대해 선형 보간
        original_indices = np.linspace(0, original_length - 1, original_length)
        target_indices = np.linspace(0, original_length - 1, target_length)
        
        resampled = np.zeros((target_length, original_samples.shape[1]), dtype=np.float32)
        for axis in range(original_samples.shape[1]):
            resampled[:, axis] = np.interp(target_indices, original_indices, original_samples[:, axis])
        
        return resampled

    def _infer_rep_samples(self) -> Tuple[Optional[int], float]:
        """
        rep 구간 샘플을 리샘플링하여 추론
        Returns: (class_id, confidence) 또는 (None, 0.0)
        """
        resampled_window = self._resample_rep_samples()
        if resampled_window is None:
            return None, 0.0
        
        # 전처리 (resampled_window는 [512, 18] 형태)
        preprocessed = self._prepare_window(resampled_window)  # [1, 512, 18]
        
        # 추론
        logits = self.backend.predict(preprocessed)[0]  # [5]
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        
        # 결과
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        
        return class_id, confidence

    def _infer(self, window: np.ndarray, timestamp: float) -> Tuple[dict, Optional[dict]]:
        """
        새로운 추론 파이프라인: 전처리 → 추론 → 스무딩 → Rep 감지
        Returns: (window_record, rep_record)
        """
        # 1. 전처리
        inputs = self._prepare_window(window)
        # 2. 모델 추론 (logits → softmax)
        logits = self.backend.predict(inputs)[0]
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        # 3. 스무딩
        class_id, is_transition = self.smoothing.update(probs)
        # 4. Rep 감지 (속도 패턴 기반)
        # 속도는 매 샘플마다 업데이트되므로 여기서는 패턴 감지만 수행
        rep_record = self.rep_detector.update(class_id, timestamp, is_transition)
        # 5. 창-레벨 레코드 생성
        p_max = float(np.max(probs))
        h = float(-np.sum(probs * np.log(probs + 1e-10)))
        window_record = {
            "timestamp": timestamp,
            "class_id": class_id,
            "p_max": p_max,
            "entropy": h,
            "probs": probs.tolist(),
            "is_transition": is_transition,
            "rep_id": self.rep_detector.rep_id if self.rep_detector.rep_start_t is not None else None,
        }
        return window_record, rep_record

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.args.host, self.args.port))
        sock.settimeout(1.0)

        if self.args.wait_for_start:
            print("\n✅ 시작 대기: 콘솔에서 Enter 또는 카메라 창에서 'S'를 누르세요.")
            if self.camera is not None:
                try:
                    self.camera.set_overlay("Press S to start (또는 콘솔 Enter)", (255, 255, 0))
                except Exception:
                    pass
            started = False
            while not started:
                if self.camera is not None and self.camera.start_event.is_set():
                    started = True
                    break
                try:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        _ = sys.stdin.readline()
                        started = True
                        break
                except Exception:
                    time.sleep(0.1)
            print("📡 스트림 수신을 시작합니다...\n")
        else:
            print("📡 스트림 수신을 즉시 시작합니다...\n")

        start_ts = time.time()
        self.session_start_ts = start_ts  # 그래프 저장용
        missed_packets = 0
        last_seq = None
        self._last_infer_step = -self.stride_samples
        _last_status_ts = start_ts
        _first_packet_received = False

        try:
            while True:
                try:
                    packet, _ = sock.recvfrom(1024)
                except socket.timeout:
                    # 주기적으로 상태 출력
                    now = time.time()
                    if now - _last_status_ts >= 2.0:
                        if not _first_packet_received:
                            print(f"[대기] 패킷 수신 대기 중... (포트 {self.args.port}, 호스트 {self.args.host})")
                        else:
                            elapsed = now - start_ts
                            hz = self.global_step / elapsed if elapsed > 0 else 0.0
                            print(f"[대기] {elapsed:6.1f}s | 샘플 {self.global_step:7d} ({hz:5.1f} Hz) | 누락 {missed_packets}")
                        _last_status_ts = now
                    continue

                try:
                    seq, millis, quality, values = decode_packet(packet)
                except (ValueError, struct.error) as e:
                    print(f"[경고] 패킷 디코딩 실패: {e}")
                    continue

                if not _first_packet_received:
                    print(f"✅ 첫 패킷 수신: seq={seq}, millis={millis}")
                    _first_packet_received = True

                if last_seq is not None and seq > last_seq + 1:
                    missed_packets += seq - (last_seq + 1)
                last_seq = seq

                if self.args.require_quality and any(q != QUALITY_OK for q in quality):
                    continue

                current_time = self.global_step / self.args.sample_rate_hz
                timestamp = start_ts + current_time

                # 그래프 저장용 데이터 기록
                self.sample_times.append(current_time)
                
                # s1 각속도 Z축 값 추출 (rep 감지용)
                # values 구조: s0(ax,ay,az,gx,gy,gz), s1(ax,ay,az,gx,gy,gz), s2(ax,ay,az,gx,gy,gz)
                # s1의 각속도 Z축: gz=values[11]
                s1_gz = float(values[11])
                # s0의 각속도 Y축: gy=values[4] (rep 검증용)
                s0_gy = float(values[4])
                
                # s1_gz 업데이트 (저역통과 필터 적용)
                filtered_gz = self.rep_detector.update_gz(s1_gz, timestamp)
                # s0_gy 업데이트 및 rep 검증
                is_rep_valid = self.rep_detector.update_s0_gy(s0_gy, timestamp)
                
                # rep 검증 실패 시 rep 취소
                if not is_rep_valid and self.rep_detector.state != self.rep_detector.STATE_IDLE:
                    print(f"\n⚠️ Rep #{self.rep_detector.rep_id} 취소: s0_gy 검증 실패 (1초 내 0.5 이상 증가 없음)")
                    self.rep_detector._reset_state()
                    self.rep_samples.clear()
                
                # 기록 (그래프용)
                self.series_s1_gz.append(filtered_gz)
                
                # 에너지 계산 (각 센서의 ax, ay, az 기반)
                # s0: values[0], values[1], values[2]
                s0_ax, s0_ay, s0_az = float(values[0]), float(values[1]), float(values[2])
                energy_s0 = np.sqrt(s0_ax**2 + s0_ay**2 + s0_az**2)
                
                # s1: values[6], values[7], values[8]
                s1_ax, s1_ay, s1_az = float(values[6]), float(values[7]), float(values[8])
                energy_s1 = np.sqrt(s1_ax**2 + s1_ay**2 + s1_az**2)
                
                # s2: values[12], values[13], values[14]
                s2_ax, s2_ay, s2_az = float(values[12]), float(values[13]), float(values[14])
                energy_s2 = np.sqrt(s2_ax**2 + s2_ay**2 + s2_az**2)
                
                # 에너지 기록
                self.series_energy_s0.append(energy_s0)
                self.series_energy_s1.append(energy_s1)
                self.series_energy_s2.append(energy_s2)
                # rep 상태 기록 (그래프용)
                self.rep_states.append(self.rep_detector.state if self.rep_detector.rep_start_t is not None else None)

                # 입력 버퍼링
                values_arr = np.asarray(values, dtype=np.float32)
                self.input_buffer.append((timestamp, values_arr))
                
                # rep 중이면 샘플 저장 (리샘플링 추론용)
                if self.rep_detector.rep_start_t is not None:
                    self.rep_samples.append((timestamp, values_arr.copy()))

                # 슬라이딩 윈도우 트리거
                if len(self.input_buffer) == self.args.window_size:
                    if self.global_step - self._last_infer_step >= self.stride_samples:
                        self._last_infer_step = self.global_step

                        window_list = [v for _, v in self.input_buffer]
                        window_np = np.stack(window_list, axis=0)

                        # 추론 및 스무딩
                        window_record, rep_record = self._infer(window_np, timestamp)
                        self.total_windows += 1

                        # 출력·로깅
                        class_id = window_record["class_id"]
                        p_max = window_record["p_max"]
                        h = window_record["entropy"]
                        is_transition = window_record.get("is_transition", False)
                        probs = np.array(window_record["probs"])
                        original_class = class_id
                        fused_class = 0 if class_id == 4 else class_id
                        
                        # Descent 과정 중에는 Knee Valgus (class 1) 판정 무시
                        if self.rep_detector.state == self.rep_detector.STATE_DESCENT and fused_class == 1:
                            # Descent 중에는 Knee Valgus를 Correct로 변경
                            fused_class = 0
                            class_id = 0  # 오버레이에도 반영
                        
                        label, feedback = CLASS_FEEDBACK[fused_class]
                        if original_class == 4:
                            feedback = "Keep going"
                        
                        # rep 진행 중인지 확인
                        is_rep_active = self.rep_detector.rep_start_t is not None
                        
                        # 현재 rep_id 업데이트 (카메라 오버레이용)
                        if is_rep_active:
                            self._current_rep_id = window_record.get("rep_id")
                        elif not is_rep_active:
                            self._current_rep_id = None
                        
                        # rep 중일 때만 상세 출력
                        if is_rep_active:
                            # TRANSITION 상태 표시
                            transition_marker = " [TRANSITION]" if is_transition else ""
                            # 각속도 정보 (디버깅용)
                            current_gz = self.rep_detector.filtered_gz
                            state_marker = f" [{self.rep_detector.state}]"
                            print(
                                f"[{millis:>9} ms] 🔎 창 #{self.total_windows} | {label}{transition_marker}{state_marker} | "
                                f"p_max={p_max:.3f} H={h:.3f} | gz={current_gz:.3f} rad/s | probs={np.round(probs, 3)}"
                            )
                            print(f"           👉 {feedback}")
                            if window_record.get("rep_id"):
                                print(f"           📍 Rep #{window_record['rep_id']} 진행 중")
                        
                        # rep_records에 새로 추가된 rep 확인 (누락된 로그 보완)
                        if len(self.rep_detector.rep_records) > self._last_logged_rep_id:
                            for new_rep in self.rep_detector.rep_records[self._last_logged_rep_id:]:
                                rep_id = new_rep.get("rep_id", 0)
                                rep_label = new_rep.get("label", "Unknown")
                                rep_conf = new_rep.get("confidence", 0.0)
                                rep_start = new_rep["start_t"] - start_ts
                                rep_end = new_rep["end_t"] - start_ts
                                rep_label_name = CLASS_LABELS_EN.get(rep_label, f"Class {rep_label}")
                                print(f"\n🎯 Rep #{rep_id} 완료: {rep_label_name} (confidence: {rep_conf:.2f})")
                                print(f"   시간: {rep_start:.2f}s~{rep_end:.2f}s, 분포: {new_rep.get('class_distribution', {})}\n")
                                self._last_logged_rep_id = rep_id
                                # 마지막 rep 정보 업데이트 (카메라 오버레이용)
                                # 실시간 기반 분류 또는 리샘플링 기반 분류 중 하나라도 Partial Squat이면 Partial Squat으로 표기
                                final_label = rep_label
                                if rep_label == 4 or new_rep.get("resampled_label") == 4:
                                    final_label = 4  # Partial Squat
                                
                                self._last_rep_info = {
                                    "rep_id": rep_id,
                                    "resampled_label": new_rep.get("resampled_label"),
                                    "final_label": final_label,  # Partial Squat 체크 포함
                                }
                        
                        # rep 완료 시에만 rep 수 출력 및 리샘플링 추론
                        if rep_record is not None:
                            rep_id = rep_record.get("rep_id", 0)
                            rep_label = rep_record.get("label", "Unknown")
                            rep_conf = rep_record.get("confidence", 0.0)
                            
                            # rep 완료 로그 출력 (항상 출력)
                            rep_start = rep_record["start_t"] - start_ts
                            rep_end = rep_record["end_t"] - start_ts
                            rep_label_name = CLASS_LABELS_EN.get(rep_label, f"Class {rep_label}")
                            
                            # rep 구간 샘플 리샘플링 추론
                            if len(self.rep_samples) > 0:
                                resampled_label, resampled_conf = self._infer_rep_samples()
                                if resampled_label is not None:
                                    rep_record["resampled_label"] = resampled_label
                                    rep_record["resampled_confidence"] = resampled_conf
                                    print(f"\n🎯 Rep #{rep_id} 완료: {rep_label_name} (confidence: {rep_conf:.2f})")
                                    print(f"   📊 리샘플링 추론: {CLASS_LABELS_EN.get(resampled_label, resampled_label)} (confidence: {resampled_conf:.2f})")
                                    print(f"   시간: {rep_start:.2f}s~{rep_end:.2f}s, 분포: {rep_record.get('class_distribution', {})}\n")
                                else:
                                    print(f"\n🎯 Rep #{rep_id} 완료: {rep_label_name} (confidence: {rep_conf:.2f})")
                                    print(f"   시간: {rep_start:.2f}s~{rep_end:.2f}s, 분포: {rep_record.get('class_distribution', {})}\n")
                            else:
                                print(f"\n🎯 Rep #{rep_id} 완료: {rep_label_name} (confidence: {rep_conf:.2f})")
                                print(f"   시간: {rep_start:.2f}s~{rep_end:.2f}s, 분포: {rep_record.get('class_distribution', {})}\n")
                            
                            # 마지막 rep 정보 저장 (카메라 오버레이용)
                            # 실시간 기반 분류 또는 리샘플링 기반 분류 중 하나라도 Partial Squat이면 Partial Squat으로 표기
                            final_label = rep_label
                            if rep_label == 4 or rep_record.get("resampled_label") == 4:
                                final_label = 4  # Partial Squat
                            
                            self._last_rep_info = {
                                "rep_id": rep_id,
                                "resampled_label": rep_record.get("resampled_label"),
                                "final_label": final_label,  # Partial Squat 체크 포함
                            }
                            
                            # rep 샘플 버퍼 초기화
                            self.rep_samples.clear()

                        # 카메라 오버레이
                        if self.camera is not None:
                            # 카메라 분류 결과 가져오기
                            _, _, cam_label = self.camera.get_camera_state()
                            
                            overlay_lines = []
                            
                            # rep 진행 중일 때만 현재 동작과 분류 표시
                            if self._current_rep_id is not None:
                                # 현재 상태 (Descent, Bottom, Ascent)
                                current_state = self.rep_detector.state
                                # 현재 분류 결과
                                current_pose = "Keep going" if original_class == 4 else CLASS_LABELS_EN.get(fused_class, f"Class {fused_class}")
                                # 상태: 분류 형식으로 표시
                                overlay_lines.append(f"{current_state}: {current_pose}")
                                overlay_lines.append(f"{self._current_rep_id}rep: ...")
                            # rep 완료 후 결과 표시
                            elif self._last_rep_info is not None:
                                rep_id = self._last_rep_info.get("rep_id", 0)
                                final_label = self._last_rep_info.get("final_label")
                                if final_label is not None:
                                    # Partial Squat 체크 포함된 최종 라벨
                                    rep_label_text = CLASS_LABELS_EN.get(final_label, f"Class {final_label}")
                                    overlay_lines.append(f"{rep_id}rep: {rep_label_text}")
                            
                            # 카메라 분류 결과를 파란색으로 하단에 표시
                            if cam_label and cam_label != "NoPerson":
                                overlay_lines.append(f"Camera: {cam_label}")
                            
                            overlay_text = "\n".join(overlay_lines) if overlay_lines else ""
                            
                            # 색상 결정: rep 진행 중이거나 완료된 경우에만 색상 적용
                            if self._current_rep_id is not None or (self._last_rep_info is not None and overlay_text):
                                overlay_color = (0, 255, 0) if (fused_class == 0 and original_class != 4) else (0, 0, 255)
                            else:
                                overlay_color = (255, 255, 255)  # 기본 색상
                            
                            self.camera.set_overlay(overlay_text, overlay_color)

                        # 창-레벨 레코드 저장
                        self.window_records.append({
                            "start_sec": current_time - (self.args.window_size / self.args.sample_rate_hz),
                            "end_sec": current_time,
                            "class_id": fused_class,
                            "confidence": p_max,
                            "entropy": h,
                        })

                        # rep-레벨 레코드 처리 (클립 저장 등)
                        if rep_record:
                            rep_id = rep_record["rep_id"]
                            
                            # rep 클립 저장
                            if self.camera is not None:
                                start_wall = rep_record["start_t"] - self.args.clip_pre_sec
                                end_wall = rep_record["end_t"] + self.args.clip_post_sec
                                clips_dir = PROJECT_ROOT / "logs/realtime_sessions" / "clips"
                                clip_path = clips_dir / f"{self.session_id}_rep_{rep_id}.mp4"
                                saved = self.camera.save_clip(start_wall, end_wall, clip_path)
                                if saved:
                                    self.clip_paths.append(saved)
                                    try:
                                        self.camera._last_clip_path = saved
                                    except Exception:
                                        pass

                self.global_step += 1
        except KeyboardInterrupt:
            print("\n🛑 사용자 중단: 소켓 종료")
        finally:
            sock.close()
            elapsed = time.time() - start_ts
            hz = self.global_step / elapsed if elapsed > 0 else 0.0
            print("\n" + "=" * 72)
            print("📊 세션 요약".center(72))
            print("=" * 72)
            print(f"총 시간:          {elapsed:8.2f} 초")
            print(f"처리 샘플 수:      {self.global_step:8d} ({hz:5.1f} Hz)")
            print(f"완료된 윈도우:     {self.total_windows:8d}")
            print(f"추정 누락 패킷:    {missed_packets:8d}")
            # 진행 중인 rep 강제 종료
            final_rep = self.rep_detector.finalize_current()
            if final_rep:
                print(f"\n🎯 Rep #{final_rep['rep_id']} 완료 (강제 종료)")
            print(f"완료된 Rep:        {len(self.rep_detector.rep_records):8d}")
            print("=" * 72)
            self._save_plot()
            if self.camera is not None:
                self.camera.stop()

    def _save_plot(self) -> None:
        """세션 종료 시 그래프 저장 (s0_az + per-window confidence)"""
        if not self.sample_times:
            print("No samples recorded; plot skipped.")
            return
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print("matplotlib is not installed; skipping plot generation.")
            return

        output_dir = PROJECT_ROOT / "logs/realtime_sessions"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"session_{self.session_id}.png"

        try:
            plt.style.use("seaborn-v0_8")
        except Exception:
            plt.style.use("seaborn")

        fig, (ax_gz, ax_energy, ax_conf) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # s1_gz (각속도 Z축) + rep 상태 구분
        ax_gz.plot(self.sample_times, self.series_s1_gz, label="s1_gz (filtered)", color="#e74c3c", linewidth=1.5)
        ax_gz.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        
        # rep 상태별 색상 표시
        if len(self.rep_states) == len(self.sample_times):
            state_colors = {
                "Idle": "#95a5a6",      # 회색
                "Descent": "#e67e22",    # 주황색
                "Bottom": "#f39c12",     # 노란색
                "Ascent": "#3498db",     # 파란색
            }
            current_state = None
            state_start_idx = 0
            for i, state in enumerate(self.rep_states):
                if state != current_state:
                    if current_state is not None and current_state in state_colors:
                        # 이전 상태 구간 색상 표시
                        ax_gz.axvspan(
                            self.sample_times[state_start_idx],
                            self.sample_times[i-1] if i > 0 else self.sample_times[0],
                            color=state_colors[current_state],
                            alpha=0.3,
                            label=current_state if state_start_idx == 0 or current_state not in [s for s in self.rep_states[:state_start_idx]] else ""
                        )
                    current_state = state
                    state_start_idx = i
            # 마지막 구간
            if current_state is not None and current_state in state_colors and state_start_idx < len(self.sample_times):
                ax_gz.axvspan(
                    self.sample_times[state_start_idx],
                    self.sample_times[-1],
                    color=state_colors[current_state],
                    alpha=0.3
                )
        ax_gz.set_ylabel("s1_gz (rad/s)")
        ax_gz.set_title("Angular Velocity Z-axis (s1_gz) - Rep Detection", fontsize=12)
        ax_gz.grid(True, axis="x", alpha=0.2)
        ax_gz.legend(loc="upper right", fontsize=8)

        # 에너지 (각 센서의 ax, ay, az 기반)
        ax_energy.plot(self.sample_times, self.series_energy_s0, label="s0 Energy", color="#e74c3c", linewidth=1.2)
        ax_energy.plot(self.sample_times, self.series_energy_s1, label="s1 Energy", color="#3498db", linewidth=1.2)
        ax_energy.plot(self.sample_times, self.series_energy_s2, label="s2 Energy", color="#2ecc71", linewidth=1.2)
        ax_energy.set_ylabel("Energy (m/s²)")
        ax_energy.set_title("Acceleration Energy (sqrt(ax²+ay²+az²))", fontsize=12)
        ax_energy.grid(True, axis="x", alpha=0.2)
        ax_energy.legend(loc="upper right", fontsize=8)

        # Confidence per window as scatter at window midpoints
        CLASS_COLORS = {
            0: "#2ecc71",
            1: "#e67e22",
            2: "#c0392b",
            3: "#f1c40f",
            4: "#3498db",
        }
        conf_times = []
        conf_values = []
        conf_colors = []
        for record in self.window_records:
            start = float(record["start_sec"])
            end = float(record["end_sec"])
            mid = (start + end) / 2.0
            conf = float(record["confidence"])
            cls = int(record["class_id"])
            conf_times.append(mid)
            conf_values.append(conf)
            conf_colors.append(CLASS_COLORS.get(cls, "#7f8c8d"))
            color = CLASS_COLORS.get(cls, "#7f8c8d")
            label = CLASS_LABELS_EN.get(cls, f"Class {cls}")
            ax_conf.axvspan(start, end, color=color, alpha=0.1)
            ax_conf.text(
                mid,
                1.02,
                label if cls != 4 else "Keep going",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                rotation=90,
                transform=ax_conf.get_xaxis_transform(),
            )
        ax_conf.scatter(conf_times, conf_values, s=18, c=conf_colors, label="confidence")
        ax_conf.set_ylim(0.0, 1.05)
        ax_conf.set_xlabel("Time (s)")
        ax_conf.set_ylabel("Confidence")
        ax_conf.set_title("Per-window confidence (with Class Labels)", fontsize=12)
        ax_conf.grid(True, axis="x", alpha=0.2)
        ax_conf.legend(loc="upper right", fontsize=8)
        
        # Rep 구간 표시 (rep 레코드 기반) - CLASS_COLORS 정의 후
        if hasattr(self, 'session_start_ts') and len(self.sample_times) > 0:
            for i, rep_record in enumerate(self.rep_detector.rep_records):
                rep_start_abs = rep_record["start_t"]
                rep_end_abs = rep_record["end_t"]
                # 절대 시간을 상대 시간으로 변환
                rep_start_rel = rep_start_abs - self.session_start_ts
                rep_end_rel = rep_end_abs - self.session_start_ts
                rep_label = rep_record.get("label", 0)
                rep_color = CLASS_COLORS.get(rep_label, "#f1c40f")
                ax_gz.axvspan(rep_start_rel, rep_end_rel, color=rep_color, alpha=0.25, edgecolor="black", linewidth=1.5, label=f"Rep #{rep_record.get('rep_id', i+1)}" if i == 0 else "")
                # rep 라벨 표시
                rep_mid = (rep_start_rel + rep_end_rel) / 2
                y_max = ax_gz.get_ylim()[1]
                ax_gz.text(rep_mid, y_max * 0.9, f"Rep #{rep_record.get('rep_id', i+1)}", 
                           ha="center", va="top", fontsize=9, fontweight="bold", 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved session plot to {output_path}")


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="새로운 실시간 스쿼트 분류 파이프라인 (EMA + FSM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("config/training_config.json"))
    parser.add_argument("--model", type=Path, default=None, help="Model file path (.tflite or .weights.h5)")
    parser.add_argument("--scaler", type=Path, default=Path("/home/jae/squat_classifier_ssl/checkpoints/squat_scaler_18axis.pkl"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--sample-rate-hz", type=float, default=110.0)
    parser.add_argument("--per-window-zscore", action="store_true", help="Apply per-window z-score")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use-keras", action="store_true", help="Use Keras + .weights.h5 instead of TFLite")
    # 슬라이딩 윈도우
    parser.add_argument("--sliding-stride-sec", type=float, default=0.25, help="슬라이딩 윈도우 stride (초). 기본값 0.25 (지연 감소)")
    # 스무딩 파이프라인
    parser.add_argument("--ema-alpha", type=float, default=0.3, help="EMA 스무딩 계수 (기본값 0.3, 스무딩 감소)")
    parser.add_argument("--uncertainty-p-max", type=float, default=0.25, help="불확실성 보류 p_max 임계값 (기본값 0.25로 더 낮춤)")
    parser.add_argument("--uncertainty-h", type=float, default=1.6, help="불확실성 보류 엔트로피 임계값 (기본값 1.6로 더 높임)")
    parser.add_argument("--majority-k", type=int, default=3, help="다수결 투표 히스토리 길이 (기본값 3, 스무딩 감소)")
    # Rep 감지 (속도 패턴 기반)
    parser.add_argument("--vel-zero-threshold", type=float, default=0.3, help="속도가 0 부근으로 간주하는 임계값 (m/s)")
    parser.add_argument("--vel-negative-threshold", type=float, default=-0.3, help="하강 구간 판단 임계값 (m/s, 기본값 -0.3, 더 민감하게)")
    parser.add_argument("--vel-positive-threshold", type=float, default=0.3, help="상승 구간 판단 임계값 (m/s, 기본값 0.3, 더 민감하게)")
    parser.add_argument("--min-rep-duration-sec", type=float, default=0.5, help="rep로 인정하기 위한 최소 지속 시간 (초, 기본값 0.5, 더 빠른 감지)")
    # 전처리
    parser.add_argument("--clip-sigma", type=float, default=6.0, help="±6σ 클리핑 값")
    # 카메라
    parser.add_argument("--enable-camera", action="store_true", help="카메라 기반 Good/Bad 보조 분류")
    parser.add_argument("--camera-model", type=Path, default=Path("/home/jae/squat_camera/squat_model/squat_model.tflite"))
    parser.add_argument("--camera-threshold", type=float, default=0.7)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=360)
    parser.add_argument("--camera-use-picamera2", action="store_true")
    parser.add_argument("--camera-rotate", type=int, choices=[0, 90, 180, 270], default=90)
    parser.add_argument("--camera-display", action="store_true", help="카메라 창 표시")
    parser.add_argument("--no-camera-draw-landmarks", action="store_false", dest="camera_draw_landmarks", default=True)
    parser.add_argument("--clip-buffer-sec", type=float, default=12.0, help="프레임 버퍼 길이")
    parser.add_argument("--clip-pre-sec", type=float, default=2.0, help="클립 저장 시 시작 지점 앞쪽 마진 (기본값 2.0초로 증가)")
    parser.add_argument("--clip-post-sec", type=float, default=0.6, help="클립 저장 시 종료 지점 뒤쪽 마진")
    # 기타
    parser.add_argument("--wait-for-start", action="store_true", default=True, help="시작 신호 대기")
    parser.add_argument("--no-wait", action="store_false", dest="wait_for_start")
    parser.add_argument("--no-quality-check", action="store_false", dest="require_quality", default=True)
    parser.add_argument("--ssid", type=str, default="ESP32_AP")
    parser.add_argument("--interface", type=str, default="wlan0")
    parser.add_argument("--skip-wifi-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> None:
    import json
    args = parse_args(argv)
    try:
        # Load config defaults
        if args.config and args.config.exists():
            cfg = json.loads(args.config.read_text(encoding="utf-8"))
        else:
            cfg = {}
        if args.window_size is None:
            args.window_size = int(cfg.get("target_len", 512))
        if args.dropout is None:
            args.dropout = float(cfg.get("dropout", 0.3))
        if not args.per_window_zscore:
            args.per_window_zscore = bool(cfg.get("per_window_zscore", True))

        # Select backend/model path defaults
        default_model_path = DEFAULT_KERAS_WEIGHTS if args.use_keras else DEFAULT_TFLITE_MODEL
        selected_model = args.model or default_model_path
        if not selected_model.exists():
            raise SystemExit(f"모델 파일을 찾을 수 없습니다: {selected_model}")
        args.model = selected_model
        if not args.scaler.exists():
            raise SystemExit(f"스케일러 파일을 찾을 수 없습니다: {args.scaler}")

        if not args.skip_wifi_check:
            try:
                ensure_esp32_connection(ssid=args.ssid, interface=args.interface)
            except Exception as exc:
                raise SystemExit(f"ESP32 AP 연결 실패: {exc}") from exc

        args.use_tflite = not args.use_keras
        # TFLite 모델 입력 길이에 맞춰 window_size 자동 동기화
        if args.use_tflite:
            try:
                interpreter = tf.lite.Interpreter(model_path=str(args.model))
                interpreter.allocate_tensors()
                in_shape = interpreter.get_input_details()[0]["shape"]
                expected_len = int(in_shape[1])
                if args.window_size != expected_len:
                    print(f"[Model] TFLite 입력 길이({expected_len})에 맞춰 window_size를 {args.window_size} -> {expected_len}로 조정합니다.")
                    args.window_size = expected_len
            except Exception:
                pass

        classifier = RealTimeClassifier(args)
        classifier.run()
    except KeyboardInterrupt:
        pass
    except SystemExit as exc:
        print(f"시스템 종료: {exc}")
        raise
    except Exception:
        tb = traceback.format_exc()
        print(f"\n에러 발생:\n{tb}")


if __name__ == "__main__":
    main(sys.argv[1:])

