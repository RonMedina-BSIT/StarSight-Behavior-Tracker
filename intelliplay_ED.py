import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import numpy as np
from collections import deque
#try
# ==========================================
# FACE LANDMARKER SETUP (MediaPipe Tasks API)
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing model file: {MODEL_PATH}\n"
        "Download it with:\n"
        "  wget -O face_landmarker.task "
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

_landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
face_landmarker = FaceLandmarker.create_from_options(_landmarker_options)


class Calibration:
    
    def __init__(self, frames_needed=30):
        self.frames_needed = frames_needed
        self.gaze_samples = []
        self.smile_samples = []
        self.done = False
        self.baseline_gaze = 0.5
        self.baseline_smile = 0.0

    def reset(self):
        self.gaze_samples.clear()
        self.smile_samples.clear()
        self.done = False

    def add_sample(self, gaze_ratio, smile_indicator):
        if self.done:
            return
        self.gaze_samples.append(gaze_ratio)
        self.smile_samples.append(smile_indicator)
        if len(self.gaze_samples) >= self.frames_needed:
        
            self.baseline_gaze = float(np.median(self.gaze_samples))
            self.baseline_smile = float(np.median(self.smile_samples))
            self.done = True


class SessionState:
 
    def __init__(self):
        self.calibration = Calibration()
        self.smile_buffer = deque(maxlen=5)
        self.gaze_buffer = deque(maxlen=5)



GAZE_AWAY_DEVIATION = 0.12      
SMILE_HAPPY_DEVIATION = 0.0035  
FRUSTRATED_DEVIATION = -0.012   
                          
EYE_HEIGHT_NORM_THRESHOLD = 0.045  


def analyze_face(image, state: SessionState):
 
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return "NO FACE DETECTED", None, None

    for landmarks in result.face_landmarks:
        # --- 1. CALCULATE ALL RAW METRICS FIRST ---

        # Distance
        left_side_x = landmarks[234].x
        right_side_x = landmarks[454].x
        face_width = max(abs(right_side_x - left_side_x), 0.0001)

        # Head Pose
        nose_x = landmarks[1].x

        # Eye Height
        left_eye_top_y = landmarks[159].y
        left_eye_bottom_y = landmarks[145].y
        right_eye_top_y = landmarks[386].y
        right_eye_bottom_y = landmarks[374].y
        average_eye_height = ((left_eye_bottom_y - left_eye_top_y) + (right_eye_bottom_y - right_eye_top_y)) / 2.0
        normalized_eye_height = average_eye_height / face_width

        # Gaze
        eye_inner_x = landmarks[133].x
        eye_outer_x = landmarks[33].x
        iris_x = landmarks[468].x
        eye_width = max(abs(eye_inner_x - eye_outer_x), 0.0001)
        raw_gaze_ratio = abs(iris_x - eye_outer_x) / eye_width
        state.gaze_buffer.append(raw_gaze_ratio)
        smoothed_gaze_ratio = sum(state.gaze_buffer) / len(state.gaze_buffer)

        # Smile
        mouth_top = landmarks[13].y
        mouth_bottom = landmarks[14].y
        mouth_left_y = landmarks[61].y
        mouth_right_y = landmarks[291].y
        mouth_center_y = (mouth_top + mouth_bottom) / 2
        raw_smile_indicator = (mouth_center_y - ((mouth_left_y + mouth_right_y) / 2)) / face_width
        state.smile_buffer.append(raw_smile_indicator)
        smoothed_smile_indicator = sum(state.smile_buffer) / len(state.smile_buffer)

        # --- 2. CALIBRATION GATE ---
        if not state.calibration.done:
            state.calibration.add_sample(smoothed_gaze_ratio, smoothed_smile_indicator)
            debug = {
                "gaze_ratio": smoothed_gaze_ratio,
                "baseline_gaze": state.calibration.baseline_gaze,
                "smile_indicator": smoothed_smile_indicator,
                "baseline_smile": state.calibration.baseline_smile,
                "eye_height_norm": normalized_eye_height,
               
                "calibration_progress": len(state.calibration.gaze_samples),
                "calibration_needed": state.calibration.frames_needed,
            }
            return "CALIBRATING - LOOK AT SCREEN", result.face_landmarks, debug

        # --- 3. CALCULATE DEVIATIONS ---
        gaze_deviation = smoothed_gaze_ratio - state.calibration.baseline_gaze
        smile_deviation = smoothed_smile_indicator - state.calibration.baseline_smile

        is_smiling = smile_deviation > SMILE_HAPPY_DEVIATION

        debug = {
            "gaze_ratio": smoothed_gaze_ratio,
            "baseline_gaze": state.calibration.baseline_gaze,
            "gaze_deviation": gaze_deviation,
            "smile_indicator": smoothed_smile_indicator,
            "baseline_smile": state.calibration.baseline_smile,
            "smile_deviation": smile_deviation,
            "eye_height_norm": normalized_eye_height,
        }

        # --- 4. DECISION TREE (The Waterfall) ---

        # A. Screen Distance Check
        if face_width < 0.15:
            return "DISTRACTED - AWAY FROM SCREEN", result.face_landmarks, debug

        # B. Head Pose Check
        if nose_x < left_side_x + 0.05 or nose_x > right_side_x - 0.05:
            return "DISTRACTED - HEAD TURNED", result.face_landmarks, debug

        # C. Eye Drooping Check (With Smiling Forgiveness!)
        dynamic_eye_threshold = EYE_HEIGHT_NORM_THRESHOLD
        if is_smiling:
            dynamic_eye_threshold = 0.020  # Much more forgiving! (Normal is 0.045)

        if normalized_eye_height < dynamic_eye_threshold:
            return "UNFOCUSED - EYES DROOPING", result.face_landmarks, debug

        # D. Gaze Check
        if abs(gaze_deviation) > GAZE_AWAY_DEVIATION:
            return "DISTRACTED - LOOKING AWAY", result.face_landmarks, debug

        # E. Emotion Check
        if is_smiling:
            label = "FOCUSED - HAPPY"
        elif smile_deviation < FRUSTRATED_DEVIATION:
            label = "FOCUSED - FRUSTRATED"
        else:
            label = "FOCUSED - NEUTRAL"

        return label, result.face_landmarks, debug

    return "NO FACE DETECTED", None, None


def draw_face_points(frame, mesh_data, color=(0, 255, 0)):
    """
    Lightweight stand-in for the old mp.solutions.drawing_utils.draw_landmarks.
    """
    h, w = frame.shape[:2]
    for landmarks in mesh_data:
        for point in landmarks:
            x, y = int(point.x * w), int(point.y * h)
            cv2.circle(frame, (x, y), 1, color, -1)


# ==========================================
# LOCAL TESTING WEBCAM LOOP
# ==========================================
if __name__ == "__main__":
    # Local demo only needs ONE session, since it's just you testing.
    demo_state = SessionState()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        emotion_label, mesh_data, debug = analyze_face(frame, demo_state)

        if "CALIBRATING" in emotion_label:
            color = (0, 255, 255)
        elif "DISTRACTED" in emotion_label or "UNFOCUSED" in emotion_label or "FRUSTRATED" in emotion_label:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        if mesh_data:
            draw_face_points(frame, mesh_data, color=(0, 255, 0))

        cv2.putText(frame, emotion_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        if debug:
            line1 = f"gaze: {debug.get('gaze_ratio', 0):.3f}  base: {debug.get('baseline_gaze', 0):.3f}  dev: {debug.get('gaze_deviation', 0):.3f}"
            line2 = f"smile: {debug.get('smile_indicator', 0):.4f}  base: {debug.get('baseline_smile', 0):.4f}  dev: {debug.get('smile_deviation', 0):.4f}"
            cv2.putText(frame, line1, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, line2, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.putText(frame, "Press 'c' to recalibrate", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow('IntelliPlay MediaPipe Monitor', frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            demo_state.calibration.reset()
            demo_state.smile_buffer.clear()
            demo_state.gaze_buffer.clear()

    cap.release()
    cv2.destroyAllWindows()