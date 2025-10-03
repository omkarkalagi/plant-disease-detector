<<<<<<< HEAD
"""
Advanced Plant Disease Detection Model Training System
Heavily optimized for maximum accuracy and performance
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D, 
    SeparableConv2D, DepthwiseConv2D, Input,
    Add, Multiply, Concatenate, Lambda
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint,
    LearningRateScheduler, CSVLogger, TensorBoard
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.applications import (
    EfficientNetB7, ResNet152V2, DenseNet201, 
    InceptionResNetV2, Xception
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPlantDiseaseDetector:
    def __init__(self, input_shape=(512, 512, 3), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.label_encoder = None
        
    def create_advanced_model(self, model_type='efficientnet'):
        """Create advanced model with multiple architecture options"""
        
        if model_type == 'efficientnet':
            return self._create_efficientnet_model()
        elif model_type == 'ensemble':
            return self._create_ensemble_model()
        elif model_type == 'custom':
            return self._create_custom_advanced_model()
        else:
            return self._create_resnet_model()
    
    def _create_efficientnet_model(self):
        """Create EfficientNet-based model"""
        base_model = EfficientNetB7(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze initial layers
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        
        # Add custom head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def _create_ensemble_model(self):
        """Create ensemble model combining multiple architectures"""
        
        # Input layer
        input_layer = Input(shape=self.input_shape)
        
        # EfficientNet branch
        effnet = EfficientNetB7(weights='imagenet', include_top=False, input_tensor=input_layer)
        effnet_out = GlobalAveragePooling2D()(effnet.output)
        effnet_out = Dense(256, activation='relu')(effnet_out)
        
        # ResNet branch
        resnet = ResNet152V2(weights='imagenet', include_top=False, input_tensor=input_layer)
        resnet_out = GlobalAveragePooling2D()(resnet.output)
        resnet_out = Dense(256, activation='relu')(resnet_out)
        
        # DenseNet branch
        densenet = DenseNet201(weights='imagenet', include_top=False, input_tensor=input_layer)
        densenet_out = GlobalAveragePooling2D()(densenet.output)
        densenet_out = Dense(256, activation='relu')(densenet_out)
        
        # Combine branches
        combined = Concatenate()([effnet_out, resnet_out, densenet_out])
        combined = BatchNormalization()(combined)
        combined = Dense(512, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(combined)
        combined = Dropout(0.5)(combined)
        combined = Dense(256, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        predictions = Dense(self.num_classes, activation='softmax')(combined)
        
        model = Model(inputs=input_layer, outputs=predictions)
        return model
    
    def _create_custom_advanced_model(self):
        """Create custom advanced CNN model"""
        
        model = Sequential([
            # Input layer
            Input(shape=self.input_shape),
            
            # First block with attention mechanism
            Conv2D(64, (7, 7), strides=2, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((3, 3), strides=2, padding='same'),
            
            # Second block
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third block with separable convolutions
            SeparableConv2D(256, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            SeparableConv2D(256, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth block
            SeparableConv2D(512, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            SeparableConv2D(512, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fifth block
            SeparableConv2D(1024, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            SeparableConv2D(1024, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dropout(0.5),
            
            # Dense layers
            Dense(1024, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01)),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(512, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_resnet_model(self):
        """Create ResNet-based model"""
        base_model = ResNet152V2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze initial layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def get_advanced_data_augmentation(self):
        """Get advanced data augmentation configuration"""
        
        return ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.3,
            shear_range=0.2,
            brightness_range=[0.6, 1.4],
            channel_shift_range=0.3,
            fill_mode='nearest',
            rescale=1./255,
            # Advanced augmentations
            featurewise_center=True,
            featurewise_std_normalization=True,
            zca_whitening=True
        )
    
    def get_advanced_callbacks(self, model_name='advanced_model'):
        """Get advanced training callbacks"""
        
        callbacks = [
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
                mode='max'
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                f'{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            ),
            
            # Learning rate scheduler
            LearningRateScheduler(
                lambda epoch: 0.001 * (0.9 ** (epoch // 10)),
                verbose=1
            ),
            
            # CSV logger
            CSVLogger(f'{model_name}_training.log'),
            
            # TensorBoard
            TensorBoard(
                log_dir=f'logs/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks
    
    def compile_model(self, model, learning_rate=0.001, optimizer='adam'):
        """Compile model with advanced settings"""
        
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate, rho=0.9)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        else:
            opt = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC()
            ]
        )
        
        return model
    
    def train_advanced_model(self, X_train, X_test, y_train, y_test, 
                           model_type='efficientnet', epochs=100, batch_size=32):
        """Train advanced model with cross-validation"""
        
        logger.info(f"Starting advanced training with {model_type} architecture")
        
        # Create model
        self.model = self.create_advanced_model(model_type)
        
        # Compile model
        self.model = self.compile_model(self.model)
        
        # Get callbacks
        callbacks = self.get_advanced_callbacks(f'advanced_{model_type}')
        
        # Get data augmentation
        datagen = self.get_advanced_data_augmentation()
        
        # Fit data augmentation
        datagen.fit(X_train)
        
        # Train model
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.model, self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get predictions
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == true_classes)
        report = classification_report(true_classes, predicted_classes, output_dict=True)
        cm = confusion_matrix(true_classes, predicted_classes)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions
        }
    
    def save_model_and_metadata(self, model_name='advanced_model'):
        """Save model and training metadata"""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Save model
        self.model.save(f'{model_name}.h5')
        
        # Save label encoder
        if self.label_encoder is not None:
            with open(f'{model_name}_label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        # Save training history
        if self.history is not None:
            with open(f'{model_name}_history.json', 'w') as f:
                json.dump(self.history.history, f, indent=2)
        
        # Save model summary
        with open(f'{model_name}_summary.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        logger.info(f"Model and metadata saved as {model_name}")
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Learning Rate
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
        
        # Top-3 Accuracy (disabled if not present)
        # Only plot if the metric exists in history
        if 'top_3_accuracy' in self.history.history:
            axes[1, 1].plot(self.history.history['top_3_accuracy'], label='Training')
            if 'val_top_3_accuracy' in self.history.history:
                axes[1, 1].plot(self.history.history['val_top_3_accuracy'], label='Validation')
            axes[1, 1].set_title('Top-3 Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Top-3 Accuracy')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved as {save_path}")

def load_dataset_advanced(dataset_path):
    """Load and preprocess dataset with advanced techniques"""
    
    images = []
    labels = []
    
    # Load images with advanced preprocessing
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    
                    # Load image
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Advanced preprocessing
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (512, 512))
                        
                        # Apply CLAHE for better contrast
                        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        lab[:,:,0] = clahe.apply(lab[:,:,0])
                        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                        
                        images.append(img)
                        labels.append(class_name)
    
    if not images:
        raise ValueError("No images found in dataset")
    
    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32) / 255.0
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    categorical_labels = tf.keras.utils.to_categorical(encoded_labels, len(label_encoder.classes_))
    
    logger.info(f"Dataset loaded: {len(images)} images, {len(label_encoder.classes_)} classes")
    
    return images, categorical_labels, label_encoder

def main():
    """Main training function"""
    
    # Load dataset
    dataset_path = "Augmented Dataset"  # or "Original Dataset"
    X, y, label_encoder = load_dataset_advanced(dataset_path)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    # Create and train model
    detector = AdvancedPlantDiseaseDetector(input_shape=(512, 512, 3), num_classes=len(label_encoder.classes_))
    detector.label_encoder = label_encoder
    
    # Train with different architectures
    architectures = ['efficientnet', 'ensemble', 'custom']
    
    for arch in architectures:
        logger.info(f"Training {arch} model...")
        
        model, history = detector.train_advanced_model(
            X_train, X_test, y_train, y_test,
            model_type=arch,
            epochs=50,
            batch_size=16
        )
        
        # Evaluate model
        results = detector.evaluate_model(X_test, y_test)
        logger.info(f"{arch} model accuracy: {results['accuracy']:.4f}")
        
        # Save model
        detector.save_model_and_metadata(f'advanced_{arch}_model')
        
        # Plot training history
        detector.plot_training_history(f'advanced_{arch}_training_history.png')

if __name__ == "__main__":
    main()
=======
"""
Advanced Plant Disease Detection Model Training System
Heavily optimized for maximum accuracy and performance
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D, 
    SeparableConv2D, DepthwiseConv2D, Input,
    Add, Multiply, Concatenate, Lambda
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint,
    LearningRateScheduler, CSVLogger, TensorBoard
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.applications import (
    EfficientNetB7, ResNet152V2, DenseNet201, 
    InceptionResNetV2, Xception
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPlantDiseaseDetector:
    def __init__(self, input_shape=(512, 512, 3), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.label_encoder = None
        
    def create_advanced_model(self, model_type='efficientnet'):
        """Create advanced model with multiple architecture options"""
        
        if model_type == 'efficientnet':
            return self._create_efficientnet_model()
        elif model_type == 'ensemble':
            return self._create_ensemble_model()
        elif model_type == 'custom':
            return self._create_custom_advanced_model()
        else:
            return self._create_resnet_model()
    
    def _create_efficientnet_model(self):
        """Create EfficientNet-based model"""
        base_model = EfficientNetB7(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze initial layers
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        
        # Add custom head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def _create_ensemble_model(self):
        """Create ensemble model combining multiple architectures"""
        
        # Input layer
        input_layer = Input(shape=self.input_shape)
        
        # EfficientNet branch
        effnet = EfficientNetB7(weights='imagenet', include_top=False, input_tensor=input_layer)
        effnet_out = GlobalAveragePooling2D()(effnet.output)
        effnet_out = Dense(256, activation='relu')(effnet_out)
        
        # ResNet branch
        resnet = ResNet152V2(weights='imagenet', include_top=False, input_tensor=input_layer)
        resnet_out = GlobalAveragePooling2D()(resnet.output)
        resnet_out = Dense(256, activation='relu')(resnet_out)
        
        # DenseNet branch
        densenet = DenseNet201(weights='imagenet', include_top=False, input_tensor=input_layer)
        densenet_out = GlobalAveragePooling2D()(densenet.output)
        densenet_out = Dense(256, activation='relu')(densenet_out)
        
        # Combine branches
        combined = Concatenate()([effnet_out, resnet_out, densenet_out])
        combined = BatchNormalization()(combined)
        combined = Dense(512, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(combined)
        combined = Dropout(0.5)(combined)
        combined = Dense(256, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        predictions = Dense(self.num_classes, activation='softmax')(combined)
        
        model = Model(inputs=input_layer, outputs=predictions)
        return model
    
    def _create_custom_advanced_model(self):
        """Create custom advanced CNN model"""
        
        model = Sequential([
            # Input layer
            Input(shape=self.input_shape),
            
            # First block with attention mechanism
            Conv2D(64, (7, 7), strides=2, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((3, 3), strides=2, padding='same'),
            
            # Second block
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third block with separable convolutions
            SeparableConv2D(256, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            SeparableConv2D(256, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth block
            SeparableConv2D(512, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            SeparableConv2D(512, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fifth block
            SeparableConv2D(1024, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            SeparableConv2D(1024, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dropout(0.5),
            
            # Dense layers
            Dense(1024, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01)),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(512, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_resnet_model(self):
        """Create ResNet-based model"""
        base_model = ResNet152V2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze initial layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def get_advanced_data_augmentation(self):
        """Get advanced data augmentation configuration"""
        
        return ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.3,
            shear_range=0.2,
            brightness_range=[0.6, 1.4],
            channel_shift_range=0.3,
            fill_mode='nearest',
            rescale=1./255,
            # Advanced augmentations
            featurewise_center=True,
            featurewise_std_normalization=True,
            zca_whitening=True
        )
    
    def get_advanced_callbacks(self, model_name='advanced_model'):
        """Get advanced training callbacks"""
        
        callbacks = [
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
                mode='max'
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                f'{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            ),
            
            # Learning rate scheduler
            LearningRateScheduler(
                lambda epoch: 0.001 * (0.9 ** (epoch // 10)),
                verbose=1
            ),
            
            # CSV logger
            CSVLogger(f'{model_name}_training.log'),
            
            # TensorBoard
            TensorBoard(
                log_dir=f'logs/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks
    
    def compile_model(self, model, learning_rate=0.001, optimizer='adam'):
        """Compile model with advanced settings"""
        
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate, rho=0.9)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        else:
            opt = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                'top_3_accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC()
            ]
        )
        
        return model
    
    def train_advanced_model(self, X_train, X_test, y_train, y_test, 
                           model_type='efficientnet', epochs=100, batch_size=32):
        """Train advanced model with cross-validation"""
        
        logger.info(f"Starting advanced training with {model_type} architecture")
        
        # Create model
        self.model = self.create_advanced_model(model_type)
        
        # Compile model
        self.model = self.compile_model(self.model)
        
        # Get callbacks
        callbacks = self.get_advanced_callbacks(f'advanced_{model_type}')
        
        # Get data augmentation
        datagen = self.get_advanced_data_augmentation()
        
        # Fit data augmentation
        datagen.fit(X_train)
        
        # Train model
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.model, self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get predictions
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == true_classes)
        report = classification_report(true_classes, predicted_classes, output_dict=True)
        cm = confusion_matrix(true_classes, predicted_classes)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions
        }
    
    def save_model_and_metadata(self, model_name='advanced_model'):
        """Save model and training metadata"""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Save model
        self.model.save(f'{model_name}.h5')
        
        # Save label encoder
        if self.label_encoder is not None:
            with open(f'{model_name}_label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        # Save training history
        if self.history is not None:
            with open(f'{model_name}_history.json', 'w') as f:
                json.dump(self.history.history, f, indent=2)
        
        # Save model summary
        with open(f'{model_name}_summary.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        logger.info(f"Model and metadata saved as {model_name}")
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Learning Rate
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
        
        # Top-3 Accuracy
        if 'top_3_accuracy' in self.history.history:
            axes[1, 1].plot(self.history.history['top_3_accuracy'], label='Training')
            axes[1, 1].plot(self.history.history['val_top_3_accuracy'], label='Validation')
            axes[1, 1].set_title('Top-3 Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Top-3 Accuracy')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved as {save_path}")

def load_dataset_advanced(dataset_path):
    """Load and preprocess dataset with advanced techniques"""
    
    images = []
    labels = []
    
    # Load images with advanced preprocessing
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    
                    # Load image
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Advanced preprocessing
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (512, 512))
                        
                        # Apply CLAHE for better contrast
                        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        lab[:,:,0] = clahe.apply(lab[:,:,0])
                        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                        
                        images.append(img)
                        labels.append(class_name)
    
    if not images:
        raise ValueError("No images found in dataset")
    
    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32) / 255.0
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    categorical_labels = tf.keras.utils.to_categorical(encoded_labels, len(label_encoder.classes_))
    
    logger.info(f"Dataset loaded: {len(images)} images, {len(label_encoder.classes_)} classes")
    
    return images, categorical_labels, label_encoder

def main():
    """Main training function"""
    
    # Load dataset
    dataset_path = "Augmented Dataset"  # or "Original Dataset"
    X, y, label_encoder = load_dataset_advanced(dataset_path)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    # Create and train model
    detector = AdvancedPlantDiseaseDetector(input_shape=(512, 512, 3), num_classes=len(label_encoder.classes_))
    detector.label_encoder = label_encoder
    
    # Train with different architectures
    architectures = ['efficientnet', 'ensemble', 'custom']
    
    for arch in architectures:
        logger.info(f"Training {arch} model...")
        
        model, history = detector.train_advanced_model(
            X_train, X_test, y_train, y_test,
            model_type=arch,
            epochs=50,
            batch_size=16
        )
        
        # Evaluate model
        results = detector.evaluate_model(X_test, y_test)
        logger.info(f"{arch} model accuracy: {results['accuracy']:.4f}")
        
        # Save model
        detector.save_model_and_metadata(f'advanced_{arch}_model')
        
        # Plot training history
        detector.plot_training_history(f'advanced_{arch}_training_history.png')

if __name__ == "__main__":
    main()
>>>>>>> e1fcd1d8ea3d427a90f7cd895c6c465448981fcb
