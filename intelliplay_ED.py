import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import numpy as np
from collections import deque

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


# ==========================================
# LANDMARK INDICES
# ==========================================
FACE_LEFT, FACE_RIGHT = 234, 454
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
NOSE_TIP = 1
FOREHEAD, CHIN = 10, 152

# Eye Aspect Ratio points: [outer corner, top1, top2, inner corner, bottom2, bottom1]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR = [263, 387, 385, 362, 380, 373]

# Gaze: (corner on the image-LEFT side, corner on the image-RIGHT side, iris center)
# Both eyes are measured left-to-right in the image so they move in the SAME direction
# when the person looks sideways (outer/inner would make them cancel when averaged).
RIGHT_EYE_GAZE = (33, 133, 468)
LEFT_EYE_GAZE = (362, 263, 473)


# ==========================================
# TUNABLE THRESHOLDS (starting points - tune with your own test clips)
# ==========================================
CALIBRATION_FRAMES = 30

# Ratios relative to each person's own calibrated baseline
FACE_WIDTH_MIN_RATIO = 0.60      # face narrower than 60% of baseline -> moved away
EAR_CLOSED_RATIO = 0.70          # eye openness below 70% of baseline counts as "closed"
SMILE_MOUTH_RATIO = 1.08         # mouth wider than 108% of baseline -> eyes may be squinting from a smile
EAR_CLOSED_RATIO_SMILING = 0.40  # stricter "closed" test while the mouth is stretched
GAZE_AWAY_DEVIATION = 0.08       # horizontal iris shift from baseline (was 0.12)
YAW_DEVIATION = 0.20             # head turned left/right
PITCH_DEVIATION = 0.10           # head tilted up/down (e.g. looking at the table)

# Time-based logic (seconds), independent of camera FPS
CLOSURE_WINDOW_S = 1.5           # look at the last 1.5 s of eye state
CLOSURE_FRACTION = 0.5           # if eyes were "closed" > 50% of that window -> drooping
CLOSURE_MIN_SAMPLES = 5

HOLD_SECONDS = {                 # how long a new state must persist before it is shown
    "FOCUSED": 0.5,
    "UNFOCUSED - EYES DROOPING": 0.3,
    "DISTRACTED - LOOKING AWAY": 1.0,
    "DISTRACTED - HEAD TURNED": 1.0,
    "DISTRACTED - AWAY FROM SCREEN": 1.0,
    "NO FACE DETECTED": 1.5,
}
DEFAULT_HOLD_S = 1.0

CALIBRATING_LABEL = "CALIBRATING - LOOK AT SCREEN"


# ==========================================
# GEOMETRY HELPERS (pixel space, so x and y are on the same scale)
# ==========================================
def _landmarks_to_pixels(landmarks, w, h):
    return np.array([(p.x * w, p.y * h) for p in landmarks], dtype=np.float32)


def _eye_aspect_ratio(pts, idx):
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = max(np.linalg.norm(p1 - p4), 1e-6)
    return float(vertical / (2.0 * horizontal))


def _iris_ratio(pts, left_i, right_i, iris_i):
    """Iris position along the eye's own axis: 0 = image-left corner, 1 = image-right corner.
    Projecting on the corner-to-corner axis keeps it correct even if the head is rolled."""
    left, right, iris = pts[left_i], pts[right_i], pts[iris_i]
    axis = right - left
    length_sq = max(float(np.dot(axis, axis)), 1e-6)
    return float(np.dot(iris - left, axis) / length_sq)


def _head_ratios(pts):
    """Yaw ratio (0.5 = facing camera) and pitch ratio of the nose within the face box."""
    left, right = pts[FACE_LEFT], pts[FACE_RIGHT]
    nose, top, chin = pts[NOSE_TIP], pts[FOREHEAD], pts[CHIN]
    face_width = max(abs(right[0] - left[0]), 1e-6)
    face_height = max(abs(chin[1] - top[1]), 1e-6)
    yaw = (nose[0] - left[0]) / face_width
    pitch = (nose[1] - top[1]) / face_height
    return float(yaw), float(pitch), float(face_width)


# ==========================================
# CALIBRATION + SESSION STATE
# ==========================================
class Calibration:
    def __init__(self, frames_needed=CALIBRATION_FRAMES):
        self.frames_needed = frames_needed
        self.done = False
        self._samples = {k: [] for k in ("gaze", "yaw", "pitch", "ear", "width", "mouth")}
        self.baseline_gaze = 0.5
        self.baseline_yaw = 0.5
        self.baseline_pitch = 0.5
        self.baseline_ear = 0.28
        self.baseline_width = 0.25
        self.baseline_mouth = 0.40

    @property
    def progress(self):
        return len(self._samples["gaze"])

    def reset(self):
        for v in self._samples.values():
            v.clear()
        self.done = False

    def add_sample(self, gaze, yaw, pitch, ear, width, mouth):
        if self.done:
            return
        # Reject bad calibration frames: head clearly turned or eyes closed (blink)
        if abs(yaw - 0.5) > 0.15 or ear < 0.15:
            return
        for key, val in zip(("gaze", "yaw", "pitch", "ear", "width", "mouth"),
                            (gaze, yaw, pitch, ear, width, mouth)):
            self._samples[key].append(val)
        if self.progress >= self.frames_needed:
            self.baseline_gaze = float(np.median(self._samples["gaze"]))
            self.baseline_yaw = float(np.median(self._samples["yaw"]))
            self.baseline_pitch = float(np.median(self._samples["pitch"]))
            self.baseline_ear = float(np.median(self._samples["ear"]))
            self.baseline_width = float(np.median(self._samples["width"]))
            self.baseline_mouth = float(np.median(self._samples["mouth"]))
            self.done = True


class SessionState:
    def __init__(self):
        self.calibration = Calibration()
        self.gaze_buffer = deque(maxlen=5)
        self.ear_buffer = deque(maxlen=5)
        self.yaw_buffer = deque(maxlen=5)
        self.pitch_buffer = deque(maxlen=5)
        self.mouth_buffer = deque(maxlen=5)
        self.closure_log = deque()          # (timestamp, eyes_closed_bool)
        self.shown_label = CALIBRATING_LABEL
        self._candidate = None
        self._candidate_since = 0.0

    def recalibrate(self):
        self.calibration.reset()
        for buf in (self.gaze_buffer, self.ear_buffer, self.yaw_buffer, self.pitch_buffer, self.mouth_buffer):
            buf.clear()
        self.closure_log.clear()
        self.shown_label = CALIBRATING_LABEL
        self._candidate = None


def _median(buf):
    return float(np.median(buf))


def _debounce(state, raw_label, now):
    """A new label is only shown after it has persisted for its hold time.
    This removes single-frame flicker (blinks, brief glances, tracking glitches)."""
    if raw_label == state.shown_label:
        state._candidate = None
        return state.shown_label

    if raw_label != state._candidate:
        state._candidate = raw_label
        state._candidate_since = now

    if now - state._candidate_since >= HOLD_SECONDS.get(raw_label, DEFAULT_HOLD_S):
        state.shown_label = raw_label
        state._candidate = None
    return state.shown_label


# ==========================================
# MAIN ANALYSIS
# ==========================================
def analyze_face(image, state: SessionState):
    now = time.monotonic()
    h, w = image.shape[:2]

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        if state.calibration.done:
            return _debounce(state, "NO FACE DETECTED", now), None, None
        return CALIBRATING_LABEL, None, None

    landmarks = result.face_landmarks[0]
    pts = _landmarks_to_pixels(landmarks, w, h)

    # --- 1. RAW METRICS ---
    yaw, pitch, face_width_px = _head_ratios(pts)
    face_width_norm = face_width_px / w  # comparable to the old 0-1 face_width

    ear = (_eye_aspect_ratio(pts, RIGHT_EYE_EAR) + _eye_aspect_ratio(pts, LEFT_EYE_EAR)) / 2.0

    gaze = (_iris_ratio(pts, *RIGHT_EYE_GAZE) + _iris_ratio(pts, *LEFT_EYE_GAZE)) / 2.0

    # Mouth width relative to face width. Only used as a guard so a smile's eye squint
    # is not mistaken for drooping (no emotion label is produced).
    mouth = float(np.linalg.norm(pts[MOUTH_RIGHT] - pts[MOUTH_LEFT]) / max(face_width_px, 1e-6))

    cal = state.calibration

    # Iris position is meaningless mid-blink, so skip it when the eyes are closing
    eyes_open_now = (not cal.done) or ear >= EAR_CLOSED_RATIO * cal.baseline_ear
    if eyes_open_now:
        state.gaze_buffer.append(gaze)
    state.ear_buffer.append(ear)
    state.yaw_buffer.append(yaw)
    state.pitch_buffer.append(pitch)
    state.mouth_buffer.append(mouth)

    # Median smoothing is more robust to landmark jitter than a plain average
    s_gaze = _median(state.gaze_buffer) if state.gaze_buffer else gaze
    s_ear = _median(state.ear_buffer)
    s_yaw = _median(state.yaw_buffer)
    s_pitch = _median(state.pitch_buffer)
    s_mouth = _median(state.mouth_buffer)

    # --- 2. CALIBRATION GATE ---
    if not cal.done:
        cal.add_sample(s_gaze, s_yaw, s_pitch, s_ear, face_width_norm, s_mouth)
        debug = {
            "gaze_ratio": s_gaze,
            "baseline_gaze": cal.baseline_gaze,
            "ear": s_ear,
            "baseline_ear": cal.baseline_ear,
            "calibration_progress": cal.progress,
            "calibration_needed": cal.frames_needed,
        }
        return CALIBRATING_LABEL, result.face_landmarks, debug

    # --- 3. DEVIATIONS FROM THIS PERSON'S BASELINE ---
    gaze_deviation = s_gaze - cal.baseline_gaze
    yaw_deviation = s_yaw - cal.baseline_yaw
    pitch_deviation = s_pitch - cal.baseline_pitch
    ear_ratio = s_ear / max(cal.baseline_ear, 1e-6)
    width_ratio = face_width_norm / max(cal.baseline_width, 1e-6)
    mouth_ratio = s_mouth / max(cal.baseline_mouth, 1e-6)
    smiling_guard = mouth_ratio > SMILE_MOUTH_RATIO
    closed_threshold = EAR_CLOSED_RATIO_SMILING if smiling_guard else EAR_CLOSED_RATIO

    debug = {
        "gaze_ratio": s_gaze,
        "baseline_gaze": cal.baseline_gaze,
        "gaze_deviation": gaze_deviation,
        "ear": s_ear,
        "baseline_ear": cal.baseline_ear,
        "ear_ratio": ear_ratio,
        "yaw_deviation": yaw_deviation,
        "pitch_deviation": pitch_deviation,
        "width_ratio": width_ratio,
        "mouth_ratio": mouth_ratio,
    }

    # --- 4. DECISION TREE (The Waterfall) ---
    raw_label = None

    # A. Head pose first (a turned head also narrows the face, so check it before distance)
    if abs(yaw_deviation) > YAW_DEVIATION or abs(pitch_deviation) > PITCH_DEVIATION:
        raw_label = "DISTRACTED - HEAD TURNED"

    # B. Screen distance (relative to where the child sat during calibration)
    elif width_ratio < FACE_WIDTH_MIN_RATIO:
        raw_label = "DISTRACTED - AWAY FROM SCREEN"

    else:
        # C. Eye drooping: fraction of recent time the eyes were mostly closed.
        # A normal blink is short, so it barely moves this fraction.
        state.closure_log.append((now, ear_ratio < closed_threshold))
        while state.closure_log and now - state.closure_log[0][0] > CLOSURE_WINDOW_S:
            state.closure_log.popleft()

        closed_fraction = sum(1 for _, c in state.closure_log if c) / len(state.closure_log)
        debug["closed_fraction"] = closed_fraction

        if len(state.closure_log) >= CLOSURE_MIN_SAMPLES and closed_fraction > CLOSURE_FRACTION:
            raw_label = "UNFOCUSED - EYES DROOPING"

        # D. Gaze direction
        elif abs(gaze_deviation) > GAZE_AWAY_DEVIATION:
            raw_label = "DISTRACTED - LOOKING AWAY"

        # E. Core engagement state
        else:
            raw_label = "FOCUSED"

    label = _debounce(state, raw_label, now)
    debug["raw_label"] = raw_label
    return label, result.face_landmarks, debug


def draw_face_points(frame, mesh_data, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    for landmarks in mesh_data:
        for point in landmarks:
            x, y = int(point.x * w), int(point.y * h)
            cv2.circle(frame, (x, y), 1, color, -1)


def _text(frame, text, org, scale, color, thickness):
    """Text with a dark outline so it stays readable on bright backgrounds."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# ==========================================
# LOCAL TESTING WEBCAM LOOP
# ==========================================
if __name__ == "__main__":
    demo_state = SessionState()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        label, mesh_data, debug = analyze_face(frame, demo_state)

        if "CALIBRATING" in label:
            color = (0, 255, 255)
        elif "DISTRACTED" in label or "UNFOCUSED" in label or "NO FACE" in label:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        if mesh_data:
            draw_face_points(frame, mesh_data, color=color)

        _text(frame, label, (20, 50), 1.0, color, 3)

        if debug:
            if "calibration_progress" in debug:
                line1 = f"calibrating: {debug['calibration_progress']}/{debug['calibration_needed']}"
                line2 = f"ear: {debug['ear']:.3f}"
            else:
                line1 = (f"gaze dev: {debug['gaze_deviation']:+.3f}  "
                         f"yaw dev: {debug['yaw_deviation']:+.3f}  "
                         f"pitch dev: {debug['pitch_deviation']:+.3f}")
                line2 = (f"ear {debug['ear_ratio']:.2f}  "
                         f"closed {debug.get('closed_fraction', 0):.2f}  "
                         f"width {debug['width_ratio']:.2f}  "
                         f"mouth {debug['mouth_ratio']:.2f}")
            _text(frame, line1, (20, 90), 0.55, (255, 255, 255), 1)
            _text(frame, line2, (20, 115), 0.55, (255, 255, 255), 1)
            if "raw_label" in debug:
                _text(frame, f"raw: {debug['raw_label']}", (20, 140), 0.55, (200, 200, 200), 1)

        cv2.putText(frame, "Press 'c' to recalibrate", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow('IntelliPlay MediaPipe Monitor', frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            demo_state.recalibrate()

    cap.release()
    cv2.destroyAllWindows()