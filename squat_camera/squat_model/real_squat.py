#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(최종 "Full" 버전)
라즈베리파이 5 + TensorFlow (Full 2.16.1) + MediaPipe
실시간 6-Feature 스쿼트 자세 분석기

[실행 환경]
이 스크립트는 "Plan Z" (황금 조합) 환경에서만 작동합니다.
- numpy<2 (1.x)
- mediapipe
- opencv-python==4.9.0.80
- tensorflow==2.16.1
- picamera2
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import time

# (중요) TFLite 런타임 대신 TensorFlow 전체를 임포트
import tensorflow as tf

# (중요) Keras의 정식 pad_sequences 함수를 임포트
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Picamera2 라이브러리
from picamera2 import Picamera2
from libcamera import controls # <--- 이 줄을 추가하세요.
# --- !!! (필수) 사용자 설정 !!! ---
# 1. Colab 학습 시 출력된 'MAX_SEQ_LENGTH' 값을 정확히 입력하세요.
#    (train_model.py 스크립트의 로그에 나옵니다)
MAX_SEQ_LENGTH = 218 # (!! 예시 값. 실제 학습 시 나온 값으로 변경하세요 !!)

# 2. 라즈베리파이에 저장한 "특수 모델"(.tflite) 파일의 *절대 경로*
#    (Colab에서 "SELECT_TF_OPS"를 켜고 변환한 99%짜리 모델)
MODEL_PATH = "/home/jae/squat_camera/squat_model/squat_model.tflite" # (!! 실제 파일 경로로 변경하세요 !!)

# 3. 카메라 해상도
CAM_WIDTH = 640
CAM_HEIGHT = 360
# --- !!! 설정 끝 !!! ---


# 분류 임계값
CLASSIFICATION_THRESHOLD = 0.7

# --- 1. 헬퍼 함수: 3점 각도 계산 ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- 2. 헬퍼 함수: 상체 기울기 계산 ---
def calculate_torso_angle(mid_shoulder, mid_hip):
    vertical_point = [mid_hip[0], mid_hip[1] - 100]
    angle = calculate_angle(mid_shoulder, mid_hip, vertical_point)
    return angle

# --- 3. (제거) Numpy Pad_Sequences 함수 (더 이상 필요 없음) ---

# --- 4. TFLite 모델 로드 (tf.lite.Interpreter 사용) ---
try:
    print(f"TFLite 모델 로드 시도: {MODEL_PATH}")
    # (수정) tflite_runtime이 아닌 tf.lite.Interpreter 사용
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH) 
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape'] 
    
    model_seq_length = input_shape[1]
    if model_seq_length != MAX_SEQ_LENGTH:
        print(f"--- [경고] ---")
        print(f"스크립트 MAX_SEQ_LENGTH ({MAX_SEQ_LENGTH})가")
        print(f"모델의 입력 길이 ({model_seq_length})와 다릅니다!")
        print(f"모델의 길이에 맞춰 {model_seq_length}로 자동 변경합니다.")
        MAX_SEQ_LENGTH = model_seq_length
    
    print(f"TFLite 모델 로드 성공. Input Shape: {input_shape}, Dtype: {input_dtype}")
    # (참고) "FlexTensorListReserve" 오류가 여기서 나지 않으면 성공!

except Exception as e:
    print(f"--- [치명적 오류] ---")
    print(f"TFLite 모델 로드 실패: {e}")
    print(f"1. MODEL_PATH 경로가 정확한지 확인하세요: {MODEL_PATH}")
    print("2. 'pip install tensorflow==2.16.1'이 올바르게 설치되었는지 확인하세요.")
    print("3. 모델이 'SELECT_TF_OPS'를 켜고 변환된 '특수 모델'이 맞는지 확인하세요.")
    raise SystemExit(e)

# --- 5. MediaPipe Pose 설정 ---
print("MediaPipe Pose 모델을 초기화합니다...")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1 
)

# --- 6. Picamera2 카메라 설정 ---
print("실시간 카메라(Picamera2)를 초기화합니다...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (CAM_WIDTH, CAM_HEIGHT)},
    lores={"size": (CAM_WIDTH, CAM_HEIGHT)} 
)
picam2.configure(config)
 
picam2.start()
print("카메라 시작 완료. 3초 후 추론을 시작합니다...")
time.sleep(3.0) 
print("--- 추론 시작 (종료하려면 'q' 키를 누르세요) ---")

# --- 7. 상태 머신 및 버퍼 변수 ---
STATE_WAITING = 0
STATE_IN_MOTION = 1
current_state = STATE_WAITING
motion_buffer = []
display_message = "Waiting"
display_color = (0, 0, 255) 
KNEE_ANGLE_THRESH_DOWN = 150.0 
KNEE_ANGLE_THRESH_UP = 155.0   

# --- 8. 메인 실시간 처리 루프 ---
frame_count = 0
try:
    while True:
        # (1) 카메라에서 프레임 캡처
        frame_bgr = picam2.capture_array("main")

        # (2) 4채널(BGRA) -> 3채널(BGR) 변환
        if frame_bgr.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)

        if frame_bgr is None:
            break

        # (3) MediaPipe 입력용 RGB 이미지 생성
        frame_count += 1
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False 

        # (4) MediaPipe 자세 추정
        results = pose.process(image_rgb)

        avg_knee_angle = -1
        torso_angle = -1
        avg_ankle_angle = -1

        # (5) 관절 추출 및 각도 계산
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr, 
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            try:
                landmarks = results.pose_landmarks.landmark
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                l_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                r_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]

                # 6개 각도 계산
                angle_l_knee = calculate_angle(l_hip, l_knee, l_ankle)
                angle_r_knee = calculate_angle(r_hip, r_knee, r_ankle)
                avg_knee_angle = (angle_l_knee + angle_r_knee) / 2.0
                angle_l_hip = calculate_angle(l_shoulder, l_hip, l_knee)
                angle_r_hip = calculate_angle(r_shoulder, r_hip, r_knee)
                mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
                mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
                angle_torso = calculate_torso_angle(mid_shoulder, mid_hip)
                torso_angle = angle_torso
                angle_l_ankle = calculate_angle(l_knee, l_ankle, l_heel)
                angle_r_ankle = calculate_angle(r_knee, r_ankle, r_heel)
                avg_ankle_angle = (angle_l_ankle + angle_r_ankle) / 2.0

                # (6) 상태 머신 로직
                current_features = [
                    angle_l_knee, angle_r_knee,
                    angle_l_hip, angle_r_hip,
                    angle_torso,
                    avg_ankle_angle
                ]

                if current_state == STATE_WAITING:
                    if avg_knee_angle < KNEE_ANGLE_THRESH_DOWN:
                        current_state = STATE_IN_MOTION
                        motion_buffer = []
                        display_message = "Recording..."
                        display_color = (0, 255, 255)

                elif current_state == STATE_IN_MOTION:
                    motion_buffer.append(current_features)

                    if avg_knee_angle > KNEE_ANGLE_THRESH_UP:
                        current_state = STATE_WAITING 
                        
                        if len(motion_buffer) > 10:
                            # --- 10. TFLite 추론 ---
                            
                            # (A) (수정) Keras의 정식 pad_sequences를 사용
                            # (중요!) Keras 함수는 리스트를 [motion_buffer]로 감싸줘야 함
                            input_data = pad_sequences([motion_buffer], 
                                                       maxlen=MAX_SEQ_LENGTH, 
                                                       dtype='float32',
                                                       padding='post', 
                                                       truncating='post')
                            
                            # (B) 모델 Dtype 맞추기
                            if input_dtype == np.float16:
                                input_data = input_data.astype(np.float16)

                            # (C) 추론 실행
                            interpreter.set_tensor(input_details[0]['index'], input_data)
                            interpreter.invoke()
                            prediction = interpreter.get_tensor(output_details[0]['index'])
                            score = prediction[0][0]

                            # (D) 결과 판별
                            if score > CLASSIFICATION_THRESHOLD:
                                display_message = f"GOOD SQUAT ({score*100:.0f}%)"
                                display_color = (0, 255, 0)
                            else:
                                display_message = f"ERROR ({score*100:.0f}%)"
                                display_color = (0, 0, 255)
                        else:
                            display_message = "Waiting"
                            display_color = (0, 0, 255)
                        
                        motion_buffer = [] # 버퍼 비우기

            except Exception as e:
                pass # 관절 일부 가려지면 통과

        # (7) 결과 시각화
        cv2.putText(frame_bgr, display_message, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, display_color, 3, cv2.LINE_AA)

        # 디버깅용 각도 표시
        cv2.putText(frame_bgr, f"Knee Angle: {avg_knee_angle:.1f}", (30, CAM_HEIGHT - 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Torso Angle: {torso_angle:.1f}", (30, CAM_HEIGHT - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # (8) 라즈베리파이 데스크톱 환경에서 실시간 창 표시
        cv2.imshow('Real-time Squat Analysis (Press q to quit)', frame_bgr)

        # (9) 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자가 'q'를 눌러 종료합니다.")
            break

except KeyboardInterrupt:
    print("\nKeyboardInterrupt 감지. 프로그램을 종료합니다.")

finally:
    # --- 12. 종료 (정리) ---
    print("정리 작업 수행...")
    picam2.stop_continuous_autofocus()
    picam2.stop()
    pose.close()
    cv2.destroyAllWindows()
    print("모든 처리가 종료되었습니다.")