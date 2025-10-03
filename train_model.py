<<<<<<< HEAD
"""
Training pipeline for Plant Disease Detection with real-time progress tracking
"""

import os
import uuid
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from config import Config
from progress_tracker import create_progress_callback

logger = logging.getLogger(__name__)


class PlantDiseaseTrainer:
    def __init__(self, app, model_builder):
        self.app = app
        self.model_builder = model_builder

    def train(self, train_dir, val_dir, epochs=10, batch_size=32, img_size=(224, 224)):
        """
        Train the plant disease detection model and update training progress.
        """
        try:
            logger.info("🚀 Starting model training...")

            # Create training session ID
            training_id = str(uuid.uuid4())
            self.app.training_sessions[training_id] = {
                'progress': {
                    'current_epoch': 0,
                    'total_epochs': epochs,
                    'loss': 0.0,
                    'accuracy': 0.0,
                    'val_loss': 0.0,
                    'val_accuracy': 0.0
                },
                'status': 'running'
            }

            # Data generators
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            val_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode='categorical'
            )

            val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode='categorical'
            )

            # Build and compile model
            model = self.model_builder.build_model(input_shape=img_size + (3,), num_classes=train_generator.num_classes)

            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
            checkpoint = ModelCheckpoint(Config.MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy')

            # Progress tracker callback
            progress_callback = create_progress_callback(training_id, self.app, epochs)

            callbacks = [early_stopping, reduce_lr, checkpoint, progress_callback]

            # Train model
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks
            )

            self.app.training_sessions[training_id]['status'] = 'completed'
            logger.info("✅ Training completed successfully")

            return training_id, history

        except Exception as e:
            logger.error(f"❌ Error during training: {e}")
            return None, None
=======
"""
Standalone script to train the plant disease classification model
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2 # Import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import Counter

from config import Config # Import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlantDiseaseClassifier:
    def __init__(self, original_dataset_path, augmented_dataset_path=None):
        self.original_dataset_path = original_dataset_path
        self.augmented_dataset_path = augmented_dataset_path
        self.model = None
        self.label_encoder = None
        self.class_names = []
        self.history = None
        self.image_count_per_class = {}
        
    def load_dataset(self, img_size=Config.IMAGE_SIZE): # Use Config.IMAGE_SIZE
        """Load and preprocess the dataset"""
        logger.info("Loading dataset...")
        
        images = []
        labels = []
        
        # Load from original dataset
        if os.path.exists(self.original_dataset_path):
            logger.info(f"Loading from original dataset: {self.original_dataset_path}")
            for class_name in os.listdir(self.original_dataset_path):
                class_path = os.path.join(self.original_dataset_path, class_name)
                if os.path.isdir(class_path):
                    logger.info(f"Loading class: {class_name}")
                    class_count = 0
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, img_file)
                            try:
                                img = cv2.imread(img_path)
                                if img is not None:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    img = cv2.resize(img, img_size)
                                    images.append(img)
                                    labels.append(class_name)
                                    class_count += 1
                            except Exception as e:
                                logger.warning(f"Error loading image {img_path}: {e}")
                    logger.info(f"Loaded {class_count} images for class {class_name}")
        
        # Load from augmented dataset if exists
        if self.augmented_dataset_path and os.path.exists(self.augmented_dataset_path):
            logger.info(f"Loading from augmented dataset: {self.augmented_dataset_path}")
            for class_name in os.listdir(self.augmented_dataset_path):
                class_path = os.path.join(self.augmented_dataset_path, class_name)
                if os.path.isdir(class_path):
                    class_count = 0
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, img_file)
                            try:
                                img = cv2.imread(img_path)
                                if img is not None:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    img = cv2.resize(img, img_size)
                                    images.append(img)
                                    labels.append(class_name)
                                    class_count += 1
                            except Exception as e:
                                logger.warning(f"Error loading image {img_path}: {e}")
                    logger.info(f"Loaded {class_count} additional images for class {class_name}")
        
        if not images:
            raise ValueError("No images found in the dataset")
        
        # Convert to numpy arrays
        images = np.array(images, dtype=np.float32) / 255.0
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.class_names = self.label_encoder.classes_.tolist()
        
        # Store class distribution
        self.image_count_per_class = Counter(labels)

        # Convert to categorical
        num_classes = len(self.label_encoder.classes_)
        categorical_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes)
        
        logger.info(f"Dataset loaded: {len(images)} images, {num_classes} classes")
        logger.info(f"Classes: {self.class_names}")
        
        return images, categorical_labels
    
    def create_model(self, input_shape, num_classes):
        """Create a transfer learning model using MobileNetV2"""
        logger.info("Creating transfer learning model with MobileNetV2...")
        
        # Load MobileNetV2 pre-trained on ImageNet, excluding the top classification layer
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=x)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, validation_split=Config.VALIDATION_SPLIT): # Use Config values
        """Train the model"""
        logger.info("Starting model training...")
        
        # Load dataset
        X, y = self.load_dataset()
        
        # Plot class distribution before training
        self.plot_class_distribution()

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, 
            stratify=np.argmax(y, axis=1)
        )
        
        logger.info(f"Training set: {len(X_train)} images")
        logger.info(f"Test set: {len(X_test)} images")
        
        # Create model
        input_shape = X_train.shape[1:]
        num_classes = y_train.shape[1]
        self.model = self.create_model(input_shape, num_classes)
        
        logger.info("Model architecture:")
        self.model.summary()
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                Config.MODEL_PATH, # Use Config.MODEL_PATH
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Generate predictions for detailed evaluation
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        report = classification_report(
            y_true_classes, y_pred_classes, 
            target_names=self.class_names
        )
        logger.info("Classification Report:")
        logger.info(f"\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        self.plot_confusion_matrix(cm)
        
        return self.history
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close() # Close plot to free memory
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close() # Close plot to free memory

    def plot_class_distribution(self):
        """Plot the distribution of images per class"""
        if not self.image_count_per_class:
            logger.warning("No class distribution data available.")
            return

        class_names = list(self.image_count_per_class.keys())
        counts = list(self.image_count_per_class.values())

        plt.figure(figsize=(12, 6))
        sns.barplot(x=class_names, y=counts, palette='viridis')
        plt.title('Distribution of Images per Class')
        plt.xlabel('Class Name')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close() # Close plot to free memory
        logger.info("Class distribution plot saved to class_distribution.png")
    
    def save_model(self, model_path=Config.MODEL_PATH, encoder_path=Config.LABEL_ENCODER_PATH): # Use Config paths
        """Save the trained model and label encoder"""
        if self.model is None:
            logger.error("No model to save. Train the model first.")
            return
        
        # Save model
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save label encoder
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"Label encoder saved to {encoder_path}")
    
    def load_model(self, model_path=Config.MODEL_PATH, encoder_path=Config.LABEL_ENCODER_PATH): # Use Config paths
        """Load a trained model and label encoder"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.class_names = self.label_encoder.classes_.tolist()
            logger.info("Model and label encoder loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, image_path):
        """Predict disease for a single image"""
        if self.model is None:
            logger.error("No model loaded. Train or load a model first.")
            return None
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, Config.IMAGE_SIZE) # Use Config.IMAGE_SIZE
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': [
                    {'class': self.class_names[i], 'confidence': float(prob)}
                    for i, prob in enumerate(predictions[0])
                ]
            }
            
        except Exception as e:
            logger.error(f"Error predicting: {e}")
            return None

def main():
    """Main training function"""
    # Paths
    original_dataset_path = Config.ORIGINAL_DATASET_PATH # Use Config path
    augmented_dataset_path = Config.AUGMENTED_DATASET_PATH # Use Config path
    
    # Create classifier
    classifier = PlantDiseaseClassifier(original_dataset_path, augmented_dataset_path)
    
    # Train model
    try:
        history = classifier.train(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE) # Use Config values
        
        # Plot training history
        classifier.plot_training_history()
        
        # Save model
        classifier.save_model()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()
>>>>>>> e1fcd1d8ea3d427a90f7cd895c6c465448981fcb
