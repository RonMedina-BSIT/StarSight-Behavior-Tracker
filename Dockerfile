# 1. Start with a lightweight Python machine
FROM python:3.9-slim

# 2. Make a working directory inside the cloud server
WORKDIR /app

# 3. CRITICAL: Install the C++ libraries that OpenCV requires to run on Linux
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements list and install the Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4b. Download the MediaPipe Face Landmarker model (Tasks API).
# Google removed the old auto-bundled `mp.solutions` API in mediapipe
# 0.10.30+, so the model now has to be fetched explicitly like this.
RUN wget -O face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# 5. Copy your actual IntelliPlay Python code into the cloud server
COPY . .

# 6. Open the port Azure will listen to
EXPOSE 8080

# 7. Turn the server on
CMD ["python", "server.py"]