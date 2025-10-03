
"""
Real-time Progress Tracker for Training
Updates training sessions with real progress data
"""

import logging
import json
from datetime import datetime
import tensorflow as tf

logger = logging.getLogger(__name__)

class ProgressTracker:
    def __init__(self, training_id, app_instance):
        self.training_id = training_id
        self.app = app_instance
        self.current_epoch = 0
        self.total_epochs = 0
        self.steps_per_epoch = 0
        self.current_batch = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch"""
        self.current_epoch = epoch + 1
        self.current_batch = 0
        self.update_progress()
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        if logs:
            self.update_progress_with_logs(logs)
            
    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        logger.info(f"Training started for {self.training_id}")
        
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        logger.info(f"Training ended for {self.training_id}")
        
    def on_batch_end(self, batch, logs=None):
        """Called at the end of each training batch to update within-epoch progress"""
        self.current_batch = batch + 1
        if hasattr(self.app, 'training_sessions') and self.training_id in self.app.training_sessions:
            progress = self.app.training_sessions[self.training_id]['progress']
            # percent across all epochs
            if self.total_epochs and self.steps_per_epoch:
                percent = ((self.current_epoch - 1) + (self.current_batch / self.steps_per_epoch)) / self.total_epochs * 100.0
            else:
                percent = 0.0
            progress['percent'] = float(percent)
            progress['current_batch'] = self.current_batch
            progress['batches_per_epoch'] = self.steps_per_epoch
            # batch-level metrics if available
            if logs:
                if 'loss' in logs:
                    progress['loss'] = float(logs.get('loss', progress.get('loss', 0.0)))
                if 'accuracy' in logs:
                    progress['accuracy'] = float(logs.get('accuracy', progress.get('accuracy', 0.0)))
            
    def update_progress(self):
        """Update progress in training sessions"""
        if hasattr(self.app, 'training_sessions') and self.training_id in self.app.training_sessions:
            progress = self.app.training_sessions[self.training_id]['progress']
            progress['current_epoch'] = self.current_epoch
            progress['total_epochs'] = self.total_epochs
            progress['batches_per_epoch'] = self.steps_per_epoch
            progress['current_batch'] = self.current_batch
            
    def update_progress_with_logs(self, logs):
        """Update progress with actual training metrics"""
        if hasattr(self.app, 'training_sessions') and self.training_id in self.app.training_sessions:
            progress = self.app.training_sessions[self.training_id]['progress']
            
            # Update with real metrics
            progress['current_epoch'] = self.current_epoch
            progress['total_epochs'] = self.total_epochs
            progress['batches_per_epoch'] = self.steps_per_epoch
            progress['current_batch'] = self.current_batch
            progress['loss'] = float(logs.get('loss', progress.get('loss', 0.0)))
            progress['accuracy'] = float(logs.get('accuracy', progress.get('accuracy', 0.0)))
            progress['val_loss'] = float(logs.get('val_loss', progress.get('val_loss', 0.0)))
            progress['val_accuracy'] = float(logs.get('val_accuracy', progress.get('val_accuracy', 0.0)))
            
            # Log progress
            logger.info(f"Epoch {self.current_epoch}/{self.total_epochs} - "
                       f"Loss: {progress['loss']:.4f}, "
                       f"Accuracy: {progress['accuracy']:.4f}, "
                       f"Val Loss: {progress['val_loss']:.4f}, "
                       f"Val Accuracy: {progress['val_accuracy']:.4f}")

def create_progress_callback(training_id, app_instance, total_epochs, steps_per_epoch=0):
    """Create a progress callback for training"""
    tracker = ProgressTracker(training_id, app_instance)
    tracker.total_epochs = total_epochs
    tracker.steps_per_epoch = steps_per_epoch
    
    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, tracker):
            super().__init__()
            self.tracker = tracker
            
        def on_train_begin(self, logs=None):
            # infer totals from Keras params if available
            try:
                if 'epochs' in self.params:
                    self.tracker.total_epochs = int(self.params.get('epochs') or self.tracker.total_epochs)
                if 'steps' in self.params:
                    self.tracker.steps_per_epoch = int(self.params.get('steps') or 0)
            except Exception:
                pass
            self.tracker.on_train_begin(logs)
            
        def on_epoch_begin(self, epoch, logs=None):
            self.tracker.on_epoch_begin(epoch, logs)
            
        def on_train_batch_end(self, batch, logs=None):
            self.tracker.on_batch_end(batch, logs)
            
        def on_epoch_end(self, epoch, logs=None):
            self.tracker.on_epoch_end(epoch, logs)
            
        def on_train_end(self, logs=None):
            self.tracker.on_train_end(logs)
    
    return ProgressCallback(tracker)
