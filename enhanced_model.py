<<<<<<< HEAD
"""
Enhanced Plant Disease Detection Model
Improved architecture with better accuracy and performance
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D, 
    SeparableConv2D, DepthwiseConv2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint,
    LearningRateScheduler
)
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import logging

logger = logging.getLogger(__name__)

def create_enhanced_model(input_shape, num_classes):
    """Create enhanced CNN model for plant disease classification"""
    
    model = Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # First Convolutional Block with Separable Convolutions
        SeparableConv2D(32, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        SeparableConv2D(32, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        SeparableConv2D(64, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        SeparableConv2D(64, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        SeparableConv2D(128, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        SeparableConv2D(128, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fourth Convolutional Block
        SeparableConv2D(256, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        SeparableConv2D(256, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fifth Convolutional Block
        SeparableConv2D(512, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        SeparableConv2D(512, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Global Average Pooling for better generalization
        GlobalAveragePooling2D(),
        
        # Dense layers with regularization
        Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_mobile_model(input_shape, num_classes):
    """Create mobile-optimized model for faster inference"""
    
    model = Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # MobileNet-style blocks
        DepthwiseConv2D(3, padding='same', depth_multiplier=1),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Conv2D(32, 1, padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        MaxPooling2D(2, 2),
        
        # Second block
        DepthwiseConv2D(3, padding='same', depth_multiplier=1),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Conv2D(64, 1, padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        MaxPooling2D(2, 2),
        
        # Third block
        DepthwiseConv2D(3, padding='same', depth_multiplier=1),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Conv2D(128, 1, padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        MaxPooling2D(2, 2),
        
        # Fourth block
        DepthwiseConv2D(3, padding='same', depth_multiplier=1),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Conv2D(256, 1, padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        MaxPooling2D(2, 2),
        
        # Global Average Pooling
        GlobalAveragePooling2D(),
        
        # Dense layers
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def get_advanced_data_augmentation():
    """Get advanced data augmentation configuration"""
    
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        shear_range=0.2,
        brightness_range=[0.7, 1.3],
        channel_shift_range=0.2,
        fill_mode='nearest',
        rescale=1./255
    )

def get_callbacks(model_name='enhanced_model'):
    """Get training callbacks for better performance"""
    
    callbacks = [
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            f'{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate scheduler
        LearningRateScheduler(
            lambda epoch: 0.001 * (0.9 ** epoch),
            verbose=1
        )
    ]
    
    return callbacks

def compile_model(model, learning_rate=0.001):
    """Compile model with optimized settings"""
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    return model

def train_enhanced_model(X_train, X_test, y_train, y_test, 
                        model_type='enhanced', epochs=50):
    """Train enhanced model with advanced techniques"""
    
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    
    # Create model
    if model_type == 'enhanced':
        model = create_enhanced_model(input_shape, num_classes)
    else:
        model = create_mobile_model(input_shape, num_classes)
    
    # Compile model
    model = compile_model(model)
    
    # Get callbacks
    callbacks = get_callbacks(f'{model_type}_plant_disease')
    
    # Get data augmentation
    datagen = get_advanced_data_augmentation()
    
    # Train model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    
    # Get predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    accuracy = np.mean(predicted_classes == true_classes)
    report = classification_report(true_classes, predicted_classes, output_dict=True)
    cm = confusion_matrix(true_classes, predicted_classes)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': predictions
    }

def save_model_and_metadata(model, label_encoder, history, model_name='enhanced_model'):
    """Save model and training metadata"""
    
    # Save model
    model.save(f'{model_name}.h5')
    
    # Save label encoder
    with open(f'{model_name}_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save training history
    import json
    with open(f'{model_name}_history.json', 'w') as f:
        json.dump(history.history, f, indent=2)
    
    logger.info(f"Model and metadata saved as {model_name}")

def load_enhanced_model(model_path='enhanced_model.h5', 
                       label_encoder_path='enhanced_model_label_encoder.pkl'):
    """Load enhanced model and label encoder"""
    
    try:
        model = tf.keras.models.load_model(model_path)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        logger.info("Enhanced model loaded successfully")
        return model, label_encoder
    except Exception as e:
        logger.error(f"Error loading enhanced model: {str(e)}")
        return None, None
=======
"""
Enhanced Plant Disease Detection Model
Improved architecture with better accuracy and performance
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D, 
    SeparableConv2D, DepthwiseConv2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint,
    LearningRateScheduler
)
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import logging

logger = logging.getLogger(__name__)

def create_enhanced_model(input_shape, num_classes):
    """Create enhanced CNN model for plant disease classification"""
    
    model = Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # First Convolutional Block with Separable Convolutions
        SeparableConv2D(32, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        SeparableConv2D(32, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        SeparableConv2D(64, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        SeparableConv2D(64, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        SeparableConv2D(128, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        SeparableConv2D(128, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fourth Convolutional Block
        SeparableConv2D(256, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        SeparableConv2D(256, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fifth Convolutional Block
        SeparableConv2D(512, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        SeparableConv2D(512, (3, 3), activation='relu', 
                       kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Global Average Pooling for better generalization
        GlobalAveragePooling2D(),
        
        # Dense layers with regularization
        Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_mobile_model(input_shape, num_classes):
    """Create mobile-optimized model for faster inference"""
    
    model = Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # MobileNet-style blocks
        DepthwiseConv2D(3, padding='same', depth_multiplier=1),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Conv2D(32, 1, padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        MaxPooling2D(2, 2),
        
        # Second block
        DepthwiseConv2D(3, padding='same', depth_multiplier=1),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Conv2D(64, 1, padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        MaxPooling2D(2, 2),
        
        # Third block
        DepthwiseConv2D(3, padding='same', depth_multiplier=1),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Conv2D(128, 1, padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        MaxPooling2D(2, 2),
        
        # Fourth block
        DepthwiseConv2D(3, padding='same', depth_multiplier=1),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Conv2D(256, 1, padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        MaxPooling2D(2, 2),
        
        # Global Average Pooling
        GlobalAveragePooling2D(),
        
        # Dense layers
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def get_advanced_data_augmentation():
    """Get advanced data augmentation configuration"""
    
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        shear_range=0.2,
        brightness_range=[0.7, 1.3],
        channel_shift_range=0.2,
        fill_mode='nearest',
        rescale=1./255
    )

def get_callbacks(model_name='enhanced_model'):
    """Get training callbacks for better performance"""
    
    callbacks = [
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            f'{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate scheduler
        LearningRateScheduler(
            lambda epoch: 0.001 * (0.9 ** epoch),
            verbose=1
        )
    ]
    
    return callbacks

def compile_model(model, learning_rate=0.001):
    """Compile model with optimized settings"""
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            'top_3_accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    return model

def train_enhanced_model(X_train, X_test, y_train, y_test, 
                        model_type='enhanced', epochs=50):
    """Train enhanced model with advanced techniques"""
    
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    
    # Create model
    if model_type == 'enhanced':
        model = create_enhanced_model(input_shape, num_classes)
    else:
        model = create_mobile_model(input_shape, num_classes)
    
    # Compile model
    model = compile_model(model)
    
    # Get callbacks
    callbacks = get_callbacks(f'{model_type}_plant_disease')
    
    # Get data augmentation
    datagen = get_advanced_data_augmentation()
    
    # Train model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    
    # Get predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    accuracy = np.mean(predicted_classes == true_classes)
    report = classification_report(true_classes, predicted_classes, output_dict=True)
    cm = confusion_matrix(true_classes, predicted_classes)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': predictions
    }

def save_model_and_metadata(model, label_encoder, history, model_name='enhanced_model'):
    """Save model and training metadata"""
    
    # Save model
    model.save(f'{model_name}.h5')
    
    # Save label encoder
    with open(f'{model_name}_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save training history
    import json
    with open(f'{model_name}_history.json', 'w') as f:
        json.dump(history.history, f, indent=2)
    
    logger.info(f"Model and metadata saved as {model_name}")

def load_enhanced_model(model_path='enhanced_model.h5', 
                       label_encoder_path='enhanced_model_label_encoder.pkl'):
    """Load enhanced model and label encoder"""
    
    try:
        model = tf.keras.models.load_model(model_path)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        logger.info("Enhanced model loaded successfully")
        return model, label_encoder
    except Exception as e:
        logger.error(f"Error loading enhanced model: {str(e)}")
        return None, None
>>>>>>> e1fcd1d8ea3d427a90f7cd895c6c465448981fcb
