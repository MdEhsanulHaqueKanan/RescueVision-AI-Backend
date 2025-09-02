# src/api/cv_api.py (The Final, Definitive, and Correct Cross-Platform Version)

import os
import sys
import cv2
import uuid
import tempfile # Import the tempfile library
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'best.pt')

try:
    model = YOLO(MODEL_PATH)
    print("--- YOLOv8 model loaded successfully ---")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load YOLOv8 model. {e}")
    model = None

# The single-image endpoint (remains the same)
@app.route('/api/detect', methods=['POST'])
def detect_objects():
    # ... (code for this function is unchanged)
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No 'image' file part."}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
    if file:
        try:
            image_bytes = file.read()
            img = Image.open(io.BytesIO(image_bytes))
            results = model(img) 
            detections = []
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    detections.append({
                        "box": [round(coord) for coord in xyxy],
                        "confidence": round(conf, 2),
                        "class_name": model.names[cls]
                    })
            return jsonify({"detections": detections})
        except Exception as e:
            import traceback
            print(f"An error occurred during detection: {e}")
            traceback.print_exc()
            return jsonify({"error": "An internal error occurred."}), 500

# The powerful video processing endpoint
@app.route('/api/process_video', methods=['POST'])
def process_video():
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500
    if 'video' not in request.files:
        return jsonify({"error": "No 'video' file part."}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected video file."}), 400
        
    # --- THE DEFINITIVE FIX ---
    # Use the tempfile library to get the correct temporary directory for the current OS.
    temp_dir = tempfile.gettempdir()
    temp_filename = f"{uuid.uuid4()}.mp4"
    temp_filepath = os.path.join(temp_dir, temp_filename)
    
    try:
        video_file.save(temp_filepath)
        print(f"Video saved temporarily to {temp_filepath}")

        cap = cv2.VideoCapture(temp_filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        all_detections = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame (e.g., every half-second)
            if fps > 0 and frame_count % int(fps / 2) == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(rgb_frame)
                timestamp_seconds = frame_count / fps
                
                for result in results:
                    for box in result.boxes:
                        if box.conf[0].item() > 0.65:
                            xyxy = box.xyxy[0].tolist()
                            all_detections.append({
                                "timestamp": round(timestamp_seconds, 2),
                                "box": [round(coord) for coord in xyxy],
                                "confidence": round(box.conf[0].item(), 2)
                            })
            frame_count += 1
        
        cap.release()
        print(f"Processing complete. Found {len(all_detections)} total detections.")
        
        return jsonify({"events": all_detections})

    except Exception as e:
        import traceback
        print(f"An error occurred during video processing: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500
    finally:
        # Crucially, ensure the temporary file is always deleted
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(f"Cleaned up temporary file: {temp_filepath}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)