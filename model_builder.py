# model_builder.py
"""
Model builder for Plant Disease Detection
Defines and compiles the deep learning model
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


class ModelBuilder:
    @staticmethod
    def build_model(input_shape=(224, 224, 3), num_classes=7):
        """
        Build and compile the plant disease detection model.
        Default: ResNet50 backbone + custom classification head
        """
        # Pre-trained ResNet50 as feature extractor
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        # Freeze base layers (transfer learning)
        for layer in base_model.layers:
            layer.trainable = False

        # Custom classification head
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        # Final model
        model = models.Model(inputs=base_model.input, outputs=outputs)

        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
