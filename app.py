from __future__ import division, print_function
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = 'model_vgg19_best.keras'
model = None
class_map = None

def get_model():
    global model
    if model is None:
        try:
            from tensorflow.keras.models import load_model
            if os.path.exists(MODEL_PATH):
                model = load_model(MODEL_PATH)
            elif os.path.exists('model_vgg19.h5'):
                model = load_model('model_vgg19.h5')
            elif os.path.exists('model_vgg19.keras'):
                model = load_model('model_vgg19.keras')
        except ImportError:
            print("ERROR: TensorFlow is not installed. Predictions will not work.")
            return None
        except Exception as e:
            print(f"ERROR: Failed to load model: {str(e)}")
            return None
    return model

def get_class_map():
    global class_map
    if class_map is None and os.path.exists('class_indices.json'):
        import json
        with open('class_indices.json','r') as f:
            class_map = json.load(f)
    return class_map

def model_predict(img_path, model):
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg19 import preprocess_input
    import numpy as np
    
    # Load and preprocess original image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x_orig = np.expand_dims(preprocess_input(x), axis=0)
    
    # Simple prediction without voting (more stable)
    preds = model.predict(x_orig)
    idx = int(np.argmax(preds, axis=1)[0])
    
    cm = get_class_map()
    if cm:
        inv = {v:k for k,v in cm.items()}
        label = inv.get(idx, str(idx))
        return "Parasitized" if label.lower().startswith("para") else "Uninfected"
    
    # Fallback to standard indices
    return "Parasitized" if idx == 0 else "Uninfected"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    print("--- POST /predict received ---")
    if 'file' not in request.files:
        print("Error: No file in request.files")
        return "No file uploaded", 400
    f = request.files['file']
    if f.filename == '':
        print("Error: Empty filename")
        return "No selected file", 400
    
    basepath = os.path.dirname(__file__)
    uploads_dir = os.path.join(basepath, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, secure_filename(f.filename))
    print(f"Saving file to: {file_path}")
    f.save(file_path)
    
    try:
        print("Loading model...")
        m = get_model()
        if m is None:
            print("Error: Model not available (TensorFlow missing?)")
            return "Model not available"
        print("Predicting...")
        result = model_predict(file_path, m)
        print(f"Prediction result: {result}")
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return f"Error: {str(e)}", 500
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print("Cleaned up file.")
            except Exception as e:
                print(f"Failed to remove file: {str(e)}")
    return result

@app.route('/gnb-stats')
def gnb_stats():
    return {
        "accuracy": 61.904761904761905,
        "precision": 66.49326805385557,
        "recall": 59.82142857142857,
        "f1": 56.50372825186412,
        "confusion_matrix": [[14, 35], [5, 51]]
    }

@app.route('/status')
def status():
    tf_available = False
    try:
        import tensorflow
        tf_available = True
    except ImportError:
        pass
    return {"status": "running", "tensorflow_available": tf_available}

@app.route('/train', methods=['POST'])
def train():
    import subprocess
    import sys
    
    # Check if dataset exists before training
    if not os.path.exists('Dataset/Train'):
        return "Error: 'Dataset/Train' folder not found. Please ensure the dataset is in the project root.", 404
        
    try:
        # Run the training script in a separate process
        subprocess.Popen([sys.executable, 'malariadetection.py'])
        return "Training started in background. The model will be saved as 'model_vgg19_best.keras' once finished."
    except Exception as e:
        return f"Error starting training: {str(e)}", 500

@app.route('/check-dataset')
def check_dataset():
    train_exists = os.path.exists('Dataset/Train')
    test_exists = os.path.exists('Dataset/Test')
    return {
        "dataset_present": train_exists and test_exists,
        "train_path": "Dataset/Train" if train_exists else "Missing",
        "test_path": "Dataset/Test" if test_exists else "Missing"
    }

@app.route('/training-status')
def training_status():
    status_file = 'training_status.json'
    if os.path.exists(status_file):
        import json
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except:
            return {"status": "error", "message": "Could not read status file"}
    return {"status": "idle", "message": "No training in progress"}

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
