
"""
Real Plant Disease Detection Model Training System
Based on latest research and best practices for 2024-2025
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import logging
import time
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealPlantDiseaseTrainer:
    def __init__(self, training_id, config):
        self.training_id = training_id
        self.config = config
        self.model = None
        self.history = None
        self.label_encoder = None
        self.training_sessions = {}
        
    def create_advanced_model(self, num_classes, input_shape=(224, 224, 3)):
        """Create an advanced model based on latest research"""
        logger.info("Creating advanced plant disease detection model...")
        
        # Use ResNet50 as base (more stable than EfficientNet)
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze initial layers, fine-tune later layers
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 30
        
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        # Add custom classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with advanced optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config.get('learning_rate', 0.001)
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model created with {model.count_params():,} parameters")
        return model
    
    def prepare_data(self, data_dir):
        """Prepare and augment training data"""
        logger.info("Preparing training data...")
        
        # Data augmentation for better generalization
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=self.config.get('batch_size', 32),
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=self.config.get('batch_size', 32),
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        class_names = sorted(train_generator.class_indices.keys())
        self.label_encoder.fit(class_names)
        
        logger.info(f"Found {len(class_names)} classes: {class_names}")
        logger.info(f"Training samples: {train_generator.samples}")
        logger.info(f"Validation samples: {val_generator.samples}")
        
        return train_generator, val_generator, class_names
    
    def train_model(self, train_generator, val_generator, class_names, app_instance=None):
        """Train the model with real progress tracking"""
        logger.info("Starting real model training...")
        
        # Create model
        self.model = self.create_advanced_model(len(class_names))
        
        # Import progress tracker
        from progress_tracker import create_progress_callback
        
        # Create callbacks for better training
        callbacks = [
            ModelCheckpoint(
                f'models/best_model_{self.training_id}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Determine training schedule
        epochs = self.config.get('epochs', 50)
        steps_per_epoch = len(train_generator)
        validation_steps = len(val_generator)

        # Add progress tracking callback if app instance is provided (after we know steps_per_epoch)
        if app_instance:
            progress_callback = create_progress_callback(self.training_id, app_instance, epochs, steps_per_epoch)
            callbacks.append(progress_callback)
        
        # Training with progress tracking
        logger.info(f"Training for {epochs} epochs...")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Validation steps: {validation_steps}")
        
        # Custom training loop for progress tracking
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed successfully!")
        return self.history
    
    def save_model(self):
        """Save trained model and label encoder"""
        logger.info("Saving trained model...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_path = f'models/plant_disease_model_{self.training_id}.h5'
        self.model.save(model_path)
        
        # Save label encoder
        encoder_path = f'models/label_encoder_{self.training_id}.pkl'
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save training history
        history_path = f'models/training_history_{self.training_id}.json'
        with open(history_path, 'w') as f:
            json.dump(self.history.history, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Label encoder saved to {encoder_path}")
        
        return model_path, encoder_path
    
    def get_training_summary(self):
        """Get training summary"""
        if self.history is None:
            return None
            
        final_accuracy = max(self.history.history['val_accuracy'])
        final_loss = min(self.history.history['val_loss'])
        
        return {
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'total_epochs': len(self.history.history['loss']),
            'best_epoch': np.argmax(self.history.history['val_accuracy']) + 1,
            'training_time': 'Real training completed',
            'model_architecture': 'EfficientNetB3 + Custom Head',
            'parameters': self.model.count_params() if self.model else 0
        }

def run_real_training(training_id, config, data_dir, app_instance=None):
    """Run real training process"""
    try:
        logger.info(f"Starting real training for {training_id}")
        
        # Initialize trainer
        trainer = RealPlantDiseaseTrainer(training_id, config)
        
        # Prepare data
        train_gen, val_gen, class_names = trainer.prepare_data(data_dir)
        
        # Train model with progress tracking
        history = trainer.train_model(train_gen, val_gen, class_names, app_instance)
        
        # Save model
        model_path, encoder_path = trainer.save_model()
        
        # Get summary
        summary = trainer.get_training_summary()
        
        logger.info(f"Real training completed for {training_id}")
        logger.info(f"Final accuracy: {summary['final_accuracy']:.4f}")
        
        return {
            'status': 'completed',
            'model_path': model_path,
            'encoder_path': encoder_path,
            'summary': summary,
            'class_names': class_names
        }
        
    except Exception as e:
        logger.error(f"Real training failed for {training_id}: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e)
        }
