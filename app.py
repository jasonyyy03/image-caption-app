# app.py - Main Flask application file
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
import torch
import open_clip
from PIL import Image
import cv2
import pickle
import base64
from io import BytesIO
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load and initialize models
def load_models():
    # Set device for CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model for image features
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L/14", pretrained="openai")
    clip_model.to(device)
    clip_model.eval()
    
    # Load tokenizer and configurations for caption model
    with open('models/coco_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open('models/coco_caption_length.pkl', 'rb') as f:
        max_caption_length = pickle.load(f)
    
    # Load embedding matrix
    embedding_matrix = np.load('models/coco_embedding_matrix.npy')
    
    # Constants for caption model
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = embedding_matrix.shape[1]
    max_objects = 10
    
    # Initialize caption model
    global_feature_dim = 768  # Adjust based on your CLIP model
    object_feature_dim = 768  # Adjust based on your CLIP model
    
    # Build and load caption model
    caption_model = build_object_enhanced_model(
        global_feature_dim=global_feature_dim,
        object_feature_dim=object_feature_dim,
        max_objects=max_objects,
        vocab_size=vocab_size,
        max_caption_length=max_caption_length,
        embedding_dim=embedding_dim,
        embedding_matrix=embedding_matrix
    )
    caption_model.load_weights('models/final_model.h5')
    
    # Load sentiment model
    sentiment_model = load_model('models/sentiment_detector.h5')
    
    # Load face cascade for sentiment detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Add alternative face detection models for improved detection
    alt_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    
    # Check if models loaded successfully
    if face_cascade.empty():
        print("WARNING: Default face cascade failed to load!")
        # Try to find it in the OpenCV installation
        cv2_prefix = os.path.dirname(os.path.abspath(cv2.__file__))
        haar_path = os.path.join(cv2_prefix, 'data/haarcascade_frontalface_default.xml')
        if os.path.exists(haar_path):
            face_cascade = cv2.CascadeClassifier(haar_path)
            print(f"Loaded cascade from OpenCV installation: {haar_path}")
    
    if alt_face_cascade.empty():
        print("WARNING: Alternative face cascade failed to load!")
        # Try to find it in the OpenCV installation
        cv2_prefix = os.path.dirname(os.path.abspath(cv2.__file__))
        alt_haar_path = os.path.join(cv2_prefix, 'data/haarcascade_frontalface_alt.xml')
        if os.path.exists(alt_haar_path):
            alt_face_cascade = cv2.CascadeClassifier(alt_haar_path)
            print(f"Loaded alt cascade from OpenCV installation: {alt_haar_path}")

    
    # Emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    return {
        'clip_model': clip_model,
        'preprocess': preprocess,
        'tokenizer': tokenizer,
        'max_caption_length': max_caption_length,
        'caption_model': caption_model,
        'sentiment_model': sentiment_model,
        'face_cascade': face_cascade,
        'alt_face_cascade': alt_face_cascade,  # Add the alternative cascade
        'emotion_labels': emotion_labels,
        'device': device
    }

# Build caption model (same as in your notebook)
def build_object_enhanced_model(global_feature_dim, object_feature_dim, max_objects, 
                       vocab_size, max_caption_length, embedding_dim, embedding_matrix):
    # Input layers
    global_input = tf.keras.layers.Input(shape=(global_feature_dim,), name='global_features')
    object_input = tf.keras.layers.Input(shape=(max_objects, object_feature_dim), name='object_features')
    caption_input = tf.keras.layers.Input(shape=(max_caption_length,), name='caption_input')
    
    # Process global features
    fe1 = tf.keras.layers.Dense(512, activation='relu')(global_input)
    fe2 = tf.keras.layers.Dense(256, activation='relu')(fe1)
    
    # Process object features
    obj_features = tf.keras.layers.GlobalAveragePooling1D()(object_input)
    obj_features = tf.keras.layers.Dense(256, activation='relu')(obj_features)
    
    # Combine visual features
    visual_context = tf.keras.layers.concatenate([fe2, obj_features])
    visual_context = tf.keras.layers.Dense(256, activation='relu')(visual_context)
    
    # Sequence processing
    se1 = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], 
                   trainable=False, mask_zero=True)(caption_input)
    se2 = tf.keras.layers.Dropout(0.4)(se1)
    se3 = tf.keras.layers.LSTM(256, return_sequences=True)(se2)
    
    # Attention mechanism
    attention = tf.keras.layers.Dot(axes=[2, 2])([tf.keras.layers.RepeatVector(max_caption_length)(visual_context), se3])
    attention = tf.keras.layers.Activation('softmax')(attention)
    context = tf.keras.layers.Lambda(lambda x: tf.einsum('bij,bik->bjk', x[0], x[1]))([attention, se3])
    context = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
    
    # Decoder
    decoder1 = tf.keras.layers.Dense(256, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.001))(tf.keras.layers.concatenate([context, visual_context]))
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder1)
    
    # Create model
    model = tf.keras.models.Model(inputs=[global_input, object_input, caption_input], outputs=outputs)
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
    return model

# Extract features using CLIP
def extract_features(image, models):
    """Extract CLIP image features for global representation"""
    preprocess = models['preprocess']
    clip_model = models['clip_model']
    device = models['device']
    
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast(device_type=device):
        feature = clip_model.encode_image(image_tensor)
        feature /= feature.norm(dim=-1, keepdim=True)  # Normalize features
    feature_array = feature.cpu().numpy()
    return feature_array

# Simplified object feature extraction
def extract_object_features(image, models, max_objects=10):
    """Create object features (simplified version)"""
    global_feature = extract_features(image, models)[0]
    feature_dim = global_feature.shape[0]
    
    # Create dummy object features with some variation
    obj_features = np.zeros((max_objects, feature_dim))
    obj_features[0] = global_feature  # First object is the global feature
    
    # Create some simulated object variations
    for i in range(1, max_objects):
        noise_scale = 0.05 * i  # Increasing variations
        obj_features[i] = global_feature * (1 + np.random.uniform(-noise_scale, noise_scale, size=feature_dim))
    
    return obj_features

# Object-aware beam search
def object_aware_beam_search(models, global_features, object_features, beam_width=5, max_steps=30):
    """Beam search for the object-enhanced model"""
    model = models['caption_model']
    tokenizer = models['tokenizer']
    max_caption_length = models['max_caption_length']
    
    # Start with startseq
    start_seq = "startseq"
    beams = [(start_seq, 0.0)]
    completed_beams = []
    
    # Ensure features are in the right shape
    if len(global_features.shape) == 1:
        global_features = global_features.reshape(1, -1)
    
    if len(object_features.shape) == 2:
        object_features = object_features.reshape(1, object_features.shape[0], -1)
    elif len(object_features.shape) == 3:
        object_features = object_features.reshape(1, object_features.shape[1], object_features.shape[2])
    
    # Loop for beam search
    for _ in range(max_steps):
        if not beams:
            break
            
        candidates = []
        
        for caption, score in beams:
            # Skip completed
            if "endseq" in caption:
                completed_beams.append((caption, score))
                continue
                
            # Process caption
            sequence = tokenizer.texts_to_sequences([caption])[0]
            sequence = pad_sequences([sequence], maxlen=max_caption_length)
            
            # Predict
            predictions = model.predict({
                'global_features': global_features,
                'object_features': object_features,
                'caption_input': sequence
            }, verbose=0)[0]
            
            # Get top predictions
            top_indices = np.argsort(predictions)[-beam_width*2:][::-1]
            
            # Add candidates
            for idx in top_indices[:beam_width]:
                word = tokenizer.index_word.get(idx, "")
                if word:
                    new_caption = f"{caption} {word}"
                    new_score = score + np.log(max(predictions[idx], 1e-10))
                    candidates.append((new_caption, new_score))
        
        # Update beams
        active_candidates = [(c, s) for c, s in candidates if "endseq" not in c]
        completed_candidates = [(c, s) for c, s in candidates if "endseq" in c]
        
        # Add to completed beams
        completed_beams.extend(completed_candidates)
        
        # Sort and keep the best beam_width
        beams = sorted(active_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Early stopping
        if len(completed_beams) >= beam_width:
            break
    
    # Get best caption
    if completed_beams:
        best_caption = sorted(completed_beams, key=lambda x: x[1], reverse=True)[0][0]
        best_caption = best_caption.replace("startseq", "").replace("endseq", "").strip()
        return best_caption
    else:
        if beams:
            best_caption = sorted(beams, key=lambda x: x[1], reverse=True)[0][0]
            best_caption = best_caption.replace("startseq", "").strip()
            return best_caption
        return "a photo of a scene"  # Fallback

# Detect and predict emotion
def predict_emotion(image, models):
    """Detect faces and predict emotions with improved accuracy and reduced false positives"""
    # Convert image to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(np.array(img_array), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply mild noise reduction while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Initialize results
    results = []
    face_cascade = models['face_cascade']
    
    # Use a more conservative approach with stricter parameters
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2,       # Higher scale factor to reduce false positives
        minNeighbors=8,        # Increase to ensure more consensus for a face
        minSize=(50, 50)       # Larger minimum size to filter out small false detections
    )
    
    # If no faces found, try slightly more sensitive parameters but still conservative
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=6,     
            minSize=(30, 30)    
        )
    
    # Apply non-maximum suppression to eliminate overlapping bounding boxes
    if len(faces) > 1:
        # Convert to format expected by NMS
        face_rects = [(x, y, x + w, y + h) for (x, y, w, h) in faces]
        
        # Calculate areas for scoring
        areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in face_rects]
        
        # Sort by area, largest first
        indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
        
        # Apply a manual non-maximum suppression
        keep = []
        for i in range(len(indices)):
            idx1 = indices[i]
            # Skip if already removed
            if idx1 not in keep and idx1 not in [j for j in range(len(indices)) if j not in keep and j != idx1]:
                keep.append(idx1)
                for j in range(i + 1, len(indices)):
                    idx2 = indices[j]
                    # Calculate intersection over union (IoU)
                    x1_1, y1_1, x2_1, y2_1 = face_rects[idx1]
                    x1_2, y1_2, x2_2, y2_2 = face_rects[idx2]
                    
                    # Calculate intersection area
                    x_left = max(x1_1, x1_2)
                    y_top = max(y1_1, y1_2)
                    x_right = min(x2_1, x2_2)
                    y_bottom = min(y2_1, y2_2)
                    
                    if x_right < x_left or y_bottom < y_top:
                        intersection_area = 0
                    else:
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    
                    # Calculate union area
                    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                    union_area = area1 + area2 - intersection_area
                    
                    # Calculate IoU
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    # If IoU is high, they're likely the same face
                    if iou > 0.3:  # threshold for considering as the same face
                        # Don't add this index to keep
                        pass
        
        # Keep only the selected face detections
        faces = [faces[i] for i in keep]
    
    # If we still have multiple faces, just keep the largest one (highest probability of being the real face)
    if len(faces) > 1:
        # Find the face with the largest area
        max_area = 0
        max_idx = 0
        for i, (x, y, w, h) in enumerate(faces):
            area = w * h
            if area > max_area:
                max_area = area
                max_idx = i
        
        # Keep only the largest face
        faces = [faces[max_idx]]
    
    # Process the remaining face(s)
    for (x, y, w, h) in faces:
        # Extract face
        face = gray[y:y+h, x:x+w]
        
        # Skip if face region is empty
        if face.size == 0:
            continue
            
        # Resize to expected input size for emotion model
        try:
            face = cv2.resize(face, (48, 48))
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            face = clahe.apply(face)
            
            # Preprocess for sentiment model
            face_input = face / 255.0
            face_input = np.expand_dims(face_input, axis=-1)
            face_input = np.expand_dims(face_input, axis=0)
            
            # Predict emotion
            prediction = models['sentiment_model'].predict(face_input, verbose=0)[0]
            emotion_idx = np.argmax(prediction)
            emotion = models['emotion_labels'][emotion_idx]
            confidence = float(prediction[emotion_idx])
            
            # Add to results
            results.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'emotion': emotion,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Error processing face: {str(e)}")
            continue
    
    return results

# Load all models during app initialization
models = None

# Initialize models at startup instead of before_first_request
with app.app_context():
    models = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    global models
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Save the image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        
        # Open image for processing
        image = Image.open(image_path).convert('RGB')
        
        # Generate caption
        global_features = extract_features(image, models)
        object_features = extract_object_features(image, models)
        caption = object_aware_beam_search(
            models=models,
            global_features=global_features[0],
            object_features=object_features
        )
        
        # Detect emotions
        emotions = predict_emotion(image, models)
        
        # Create image with bounding boxes for response
        img_array = np.array(image)
        img_cv = cv2.cvtColor(np.array(img_array), cv2.COLOR_RGB2BGR)
        
        for face in emotions:
            x, y, w, h = face['bbox']
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{face['emotion']} ({face['confidence']:.2f})"
            cv2.putText(img_cv, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert back to PIL and then to base64 for response
        img_with_faces = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        img_with_faces.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Return results
        return jsonify({
            'caption': caption,
            'emotions': emotions,
            'image': img_str
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)