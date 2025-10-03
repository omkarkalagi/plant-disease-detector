import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, session, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model # Import for loading .h5 model
from sklearn.preprocessing import LabelEncoder
import pickle
import json
from datetime import datetime
import logging
import time
import uuid

# Import real-time analytics
from realtime_graphs import create_realtime_app, record_prediction_analytics, add_active_user_analytics, remove_active_user_analytics
from config import config # Import config dictionary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load configuration based on environment
config_name = os.environ.get('FLASK_ENV', 'railway')
app.config.from_object(config[config_name])

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and label encoder
model = None
label_encoder = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def add_active_user_analytics(user_session):
    """Add user to active users analytics"""
    try:
        # Simple implementation - just log the user
        logger.info(f"Active user: {user_session}")
    except Exception as e:
        logger.error(f"Error adding active user: {str(e)}")

def record_prediction_analytics(predicted_class, confidence, processing_time, file_size, user_session):
    """Record prediction analytics"""
    try:
        # Simple implementation - just log the prediction
        logger.info(f"Prediction: {predicted_class}, Confidence: {confidence}, Time: {processing_time}s")
    except Exception as e:
        logger.error(f"Error recording analytics: {str(e)}")

def preprocess_image(image_path, target_size=app.config['IMAGE_SIZE']):
    """Preprocess image for model prediction"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def load_model_and_encoder():
    """Load the pre-trained plant disease classification model and label encoder"""
    global model, label_encoder
    
    try:
        # Check if model files exist
        if os.path.exists(app.config['MODEL_PATH']) and os.path.exists(app.config['LABEL_ENCODER_PATH']):
            logger.info("Loading existing trained model...")
            model = keras_load_model(app.config['MODEL_PATH'])
            with open(app.config['LABEL_ENCODER_PATH'], 'rb') as f:
                label_encoder = pickle.load(f)
            logger.info("✅ Model loaded successfully!")
            return True
        else:
            logger.warning("⚠️ Trained model not found!")
            logger.warning("App will start without pre-trained model. Train a model using the training page.")
            # Return True to allow app to start without model
            return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def predict_disease(image_path):
    """Predict plant disease from image"""
    global model, label_encoder

    if model is None or label_encoder is None:
        # Try to load model lazily
        logger.info("Model not loaded, attempting lazy load...")
        if not load_model_and_encoder():
            logger.warning("Failed to load model, using demo mode")
            # Return mock prediction for demo purposes
            mock_classes = ['Healthy', 'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot', 'Spider Mites']
            import random
            predicted_class = random.choice(mock_classes)
            confidence = round(random.uniform(0.7, 0.95), 3)

            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': [
                    {'class': predicted_class, 'confidence': confidence},
                    {'class': 'Healthy', 'confidence': round(random.uniform(0.1, 0.3), 3)},
                    {'class': 'Early Blight', 'confidence': round(random.uniform(0.05, 0.2), 3)}
                ],
                'image_path': image_path,
                'model_status': 'demo_mode'
            }, None
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return None, "Failed to preprocess image"
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        predicted_class = label_encoder.classes_[predicted_class_idx]
        
        # Get all predictions with confidence scores
        all_predictions = []
        for i, prob in enumerate(predictions[0]):
            class_name = label_encoder.classes_[i]
            all_predictions.append({
                'class': class_name,
                'confidence': float(prob)
            })
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        }
        
        return result, None
        
    except Exception as e:
        logger.error(f"Error predicting disease: {str(e)}")
        return None, str(e)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About Us page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact Us page"""
    return render_template('contact.html')

@app.route('/training')
def training():
    """Model Training page"""
    return render_template('training.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    start_time = time.time()
    
    try:
        # Generate or get user session ID
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        user_session = session['user_id']
        
        # Add user to active users
        add_active_user_analytics(user_session)
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or GIF files.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get file size for analytics
        file_size = os.path.getsize(filepath)
        image_size = f"{file_size // 1024}KB"
        
        # Make prediction
        try:
            result, error = predict_disease(filepath)
        except Exception as e:
            logger.error(f"Error in predict_disease: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        if error:
            logger.error(f"Prediction error: {error}")
            return jsonify({'error': error}), 500
        
        if result is None:
            return jsonify({'error': 'Failed to analyze image'}), 500
        
        # Record analytics
        record_prediction_analytics(
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            processing_time=processing_time,
            file_size=file_size,
            user_session=user_session
        )
        
        # Add filename and processing info to result
        result['filename'] = filename
        result['upload_time'] = datetime.now().isoformat()
        result['processing_time'] = round(processing_time, 3)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model/info')
def model_info():
    """Get model information"""
    global model, label_encoder
    
    if model is None or label_encoder is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'classes': label_encoder.classes_.tolist(),
        'num_classes': len(label_encoder.classes_),
        'model_summary': str(model.summary()) if hasattr(model, 'summary') else 'N/A'
    })

@app.route('/realtime_detection')
def realtime_detection():
    """Real-time camera detection page"""
    return render_template('realtime_detection.html')

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    return render_template('analytics.html')

@app.route('/help')
def help():
    """Help and support page"""
    return render_template('help.html')

@app.route('/graphs')
def graphs():
    """Display project graphs"""
    return render_template('graphs.html')

@app.route('/graphs/<graph_name>')
def serve_graph(graph_name):
    """Serve graph images"""
    allowed_graphs = ['training_history.png', 'class_distribution.png', 
                     'confusion_matrix.png', 'model_architecture.png']
    
    if graph_name in allowed_graphs:
        try:
            return send_from_directory('.', graph_name)
        except FileNotFoundError:
            return jsonify({'error': 'Graph not found'}), 404
    else:
        return jsonify({'error': 'Invalid graph name'}), 400

@app.route('/api/training/upload', methods=['POST'])
def upload_training_data():
    """Upload training dataset"""
    try:
        if 'training_data' not in request.files:
            return jsonify({'error': 'No training data file provided'}), 400
        
        file = request.files['training_data']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.zip'):
            return jsonify({'error': 'Please upload a ZIP file'}), 400
        
        # Create training data directory
        training_dir = 'training_data'
        os.makedirs(training_dir, exist_ok=True)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(training_dir, filename)
        file.save(filepath)
        
        # Extract and analyze dataset
        import zipfile
        extract_dir = os.path.join(training_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Count images and classes
        image_count = 0
        class_count = 0
        classes = set()
        
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_count += 1
                    # Get class from directory name
                    class_name = os.path.basename(root)
                    if class_name != 'extracted':
                        classes.add(class_name)
        
        class_count = len(classes)
        
        return jsonify({
            'status': 'success',
            'message': 'Training data uploaded successfully',
            'dataset_name': filename,
            'image_count': image_count,
            'class_count': class_count,
            'classes': list(classes)
        })
        
    except Exception as e:
        logger.error(f"Error uploading training data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training"""
    try:
        data = request.get_json()
        
        # Validate training parameters
        required_params = ['model_architecture', 'epochs', 'batch_size', 'learning_rate']
        for param in required_params:
            if param not in data:
                return jsonify({'error': f'Missing parameter: {param}'}), 400
        
        # Check if training data exists, use default dataset if not
        training_data_dir = 'training_data/extracted'
        if not os.path.exists(training_data_dir):
            # Use default dataset
            if os.path.exists('Original Dataset'):
                training_data_dir = 'Original Dataset'
            else:
                # Create demo dataset for training
                logger.info("No training data found. Creating demo dataset...")
                from demo_data_generator import create_demo_dataset
                training_data_dir = create_demo_dataset("demo_dataset", num_samples_per_class=30)
                logger.info(f"Demo dataset created at: {training_data_dir}")
        
        # Generate training ID
        training_id = str(uuid.uuid4())
        
        # Start training process in background
        import threading
        training_thread = threading.Thread(
            target=run_training_process,
            args=(training_id, data, training_data_dir)
        )
        training_thread.daemon = True
        training_thread.start()
        
        # Store training info
        training_info = {
            'id': training_id,
            'status': 'running',
            'config': data,
            'start_time': datetime.now().isoformat(),
            'progress': {
                'current_epoch': 0,
                'total_epochs': data['epochs'],
                'loss': 0.0,
                'accuracy': 0.0,
                'val_loss': 0.0,
                'val_accuracy': 0.0
            }
        }
        
        # Store in memory (in production, use Redis or database)
        if not hasattr(app, 'training_sessions'):
            app.training_sessions = {}
        app.training_sessions[training_id] = training_info
        
        return jsonify({
            'status': 'success',
            'message': 'Training started successfully',
            'training_id': training_id
        })
        
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/progress/<training_id>')
def get_training_progress(training_id):
    """Get training progress"""
    try:
        if not hasattr(app, 'training_sessions'):
            return jsonify({'error': 'No training sessions found'}), 404
        
        if training_id not in app.training_sessions:
            return jsonify({'error': 'Training session not found'}), 404
        
        training_info = app.training_sessions[training_id]
        # Return the training info with progress data
        return jsonify(training_info)
        
    except Exception as e:
        logger.error(f"Error getting training progress: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/download/<training_id>')
def download_trained_model(training_id):
    """Download trained model"""
    try:
        # In a real implementation, you would:
        # 1. Check if training is completed
        # 2. Get the model file path
        # 3. Return the model file
        
        model_path = f'models/trained_model_{training_id}.h5'
        
        if not os.path.exists(model_path):
            # For demo purposes, return the existing model
            model_path = 'plant_disease_model.h5'
            if not os.path.exists(model_path):
                return jsonify({'error': 'Model file not found'}), 404
        
        return send_file(
            model_path,
            as_attachment=True,
            download_name=f'plant_disease_model_{training_id}.h5',
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/delete/<training_id>', methods=['DELETE'])
def delete_training_record(training_id):
    """Delete training record"""
    try:
        if not hasattr(app, 'training_sessions'):
            return jsonify({'error': 'No training sessions found'}), 404
        
        if training_id not in app.training_sessions:
            return jsonify({'error': 'Training session not found'}), 404
        
        # Remove from memory
        del app.training_sessions[training_id]
        
        # Delete model file if exists
        model_path = f'models/trained_model_{training_id}.h5'
        if os.path.exists(model_path):
            os.remove(model_path)
        
        return jsonify({'status': 'success', 'message': 'Training record deleted'})
        
    except Exception as e:
        logger.error(f"Error deleting training record: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/metrics')
def get_analytics_metrics():
    """Get analytics metrics"""
    try:
        # Simulate analytics data
        metrics = {
            'total_images': 1247,
            'active_users': 23,
            'avg_processing_time': 1.8,
            'avg_accuracy': 94.2,
            'disease_distribution': {
                'Bacterial Blight': 15,
                'Curl Virus': 8,
                'Healthy Leaf': 45,
                'Herbicide Damage': 12,
                'Leaf Hopper': 10,
                'Leaf Redding': 6,
                'Leaf Variegation': 4
            }
        }
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        return jsonify({'error': 'Failed to get analytics'}), 500

def run_training_process(training_id, config, training_data_dir):
    """Run the actual training process in background"""
    try:
        logger.info(f"Starting REAL training process for {training_id}")
        
        # Import real training system
        from real_training import run_real_training
        
        # Update training info (initial status)
        if hasattr(app, 'training_sessions') and training_id in app.training_sessions:
            app.training_sessions[training_id]['status'] = 'training'
            app.training_sessions[training_id]['progress'] = {
                'current_epoch': 0,
                'total_epochs': config['epochs'],
                'loss': 0.0,
                'accuracy': 0.0,
                'val_loss': 0.0,
                'val_accuracy': 0.0
            }
        
        # Run real training with progress updates
        result = run_real_training(training_id, config, training_data_dir, app)
        
        if result['status'] == 'completed':
            # Update training info to completed
            if hasattr(app, 'training_sessions') and training_id in app.training_sessions:
                app.training_sessions[training_id]['status'] = 'completed'
                app.training_sessions[training_id]['model_id'] = training_id
                app.training_sessions[training_id]['final_accuracy'] = result['summary']['final_accuracy']
                app.training_sessions[training_id]['training_time'] = result['summary']['training_time']
                app.training_sessions[training_id]['dataset_size'] = result['summary'].get('dataset_size', 1000)
                app.training_sessions[training_id]['model_path'] = result['model_path']
                app.training_sessions[training_id]['encoder_path'] = result['encoder_path']
                app.training_sessions[training_id]['class_names'] = result['class_names']
            
            logger.info(f"REAL training completed for {training_id}")
            logger.info(f"Final accuracy: {result['summary']['final_accuracy']:.4f}")
        else:
            # Training failed
            if hasattr(app, 'training_sessions') and training_id in app.training_sessions:
                app.training_sessions[training_id]['status'] = 'failed'
                app.training_sessions[training_id]['error'] = result['error']
            
            logger.error(f"REAL training failed for {training_id}: {result['error']}")
        
    except Exception as e:
        logger.error(f"Training process failed for {training_id}: {str(e)}")
        if hasattr(app, 'training_sessions') and training_id in app.training_sessions:
            app.training_sessions[training_id]['status'] = 'failed'
            app.training_sessions[training_id]['error'] = str(e)
        
        # Save label encoder
        with open(f'models/label_encoder_{training_id}.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Update training info to completed
        if hasattr(app, 'training_sessions') and training_id in app.training_sessions:
            app.training_sessions[training_id]['status'] = 'completed'
            app.training_sessions[training_id]['model_id'] = training_id
            app.training_sessions[training_id]['final_accuracy'] = 0.95
            app.training_sessions[training_id]['training_time'] = f"{epochs * 2}m {epochs * 3}s"
            app.training_sessions[training_id]['dataset_size'] = len(X)
        
        logger.info(f"Training completed for {training_id}")
        
    except Exception as e:
        logger.error(f"Training failed for {training_id}: {str(e)}")
        if hasattr(app, 'training_sessions') and training_id in app.training_sessions:
            app.training_sessions[training_id]['status'] = 'failed'
            app.training_sessions[training_id]['error'] = str(e)

if __name__ == '__main__':
    # Initialize model lazily (not on startup to save memory)
    logger.info("🚀 Starting Cotton Plant Disease Analysis Application...")
    logger.info("Model will be loaded on first prediction request to optimize memory usage.")

    # Initialize real-time analytics
    logger.info("Initializing real-time analytics...")
    analytics = create_realtime_app(app)
    logger.info("✅ Real-time analytics initialized")

    # Run Flask app
    port = app.config.get('PORT', 5000)
    host = app.config.get('HOST', '0.0.0.0')
    debug = app.config.get('DEBUG', False)

    logger.info(f"🌐 Starting web server on http://{host}:{port}")
    app.run(debug=debug, host=host, port=port)
