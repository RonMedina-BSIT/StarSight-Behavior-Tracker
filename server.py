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
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    session_id = request.form.get('session_id', 'default')
    state = get_state(session_id)

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Ito ang mismong fix: Ipinapasa na natin ang frame at ang state
    emotion_label, _, debug = intelliplay_ED.analyze_face(frame, state)

    response = {
        "status": "success",
        "label": emotion_label, 
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