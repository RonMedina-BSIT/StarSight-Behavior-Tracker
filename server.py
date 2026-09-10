from flask import Flask, request, jsonify
import cv2
import numpy as np

import intelliplay_ED

app = Flask(__name__)


sessions = {}


def get_state(session_id):
    if session_id not in sessions:
        sessions[session_id] = intelliplay_ED.SessionState()
    return sessions[session_id]


@app.route('/analyze', methods=['POST'])
def analyze_frame():
    # 1. Check if the Flutter app actually sent an image
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    
    session_id = request.form.get('session_id', 'default')
    state = get_state(session_id)

    file = request.files['image']

    # 2. Convert the incoming web image into an OpenCV format
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # 3. Send the image + this session's own state to the analysis function
    emotion_label, _, debug = intelliplay_ED.analyze_face(frame, state)

    # 4. Send the result back to Flutter as a JSON response
    response = {
        "status": "success",
        "emotion": emotion_label,
    }
    
    if debug and "calibration_progress" in debug:
        response["calibration_progress"] = debug["calibration_progress"]
        response["calibration_needed"] = debug["calibration_needed"]

    return jsonify(response)


@app.route('/reset_calibration', methods=['POST'])
def reset_calibration():
 
    session_id = request.form.get('session_id', 'default')
    state = get_state(session_id)
    state.calibration.reset()
    state.smile_buffer.clear()
    state.gaze_buffer.clear()
    return jsonify({"status": "reset"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)