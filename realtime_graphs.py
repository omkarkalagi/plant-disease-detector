"""
Real-time graphs and analytics system for Plant Disease Analysis
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
import threading
import time
import random
import numpy as np
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeAnalytics:
    def __init__(self, db_path='analytics.db'):
        self.db_path = db_path
        self.init_database()
        
        # In-memory storage for real-time data
        self.recent_predictions = deque(maxlen=100)  # Last 100 predictions
        self.active_users = set()
        self.prediction_counts = defaultdict(int)
        self.confidence_scores = deque(maxlen=50)
        self.hourly_stats = defaultdict(int)
        
        # Performance metrics
        self.total_predictions = 0
        self.successful_predictions = 0
        self.average_confidence = 0.0
        self.processing_times = deque(maxlen=20)
        
        # Start background tasks
        self.start_background_tasks()
    
    def init_database(self):
        """Initialize SQLite database for persistent analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                predicted_class TEXT,
                confidence REAL,
                processing_time REAL,
                user_session TEXT,
                image_size TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date DATE PRIMARY KEY,
                total_predictions INTEGER DEFAULT 0,
                unique_users INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.0,
                most_common_disease TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_usage (
                hour INTEGER PRIMARY KEY,
                prediction_count INTEGER DEFAULT 0,
                avg_processing_time REAL DEFAULT 0.0
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Analytics database initialized")
    
    def record_prediction(self, predicted_class, confidence, processing_time, user_session, image_size=None):
        """Record a new prediction for analytics"""
        try:
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (predicted_class, confidence, processing_time, user_session, image_size)
                VALUES (?, ?, ?, ?, ?)
            ''', (predicted_class, confidence, processing_time, user_session, image_size))
            conn.commit()
            conn.close()
            
            # Update in-memory data
            prediction_data = {
                'timestamp': datetime.now().isoformat(),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'processing_time': processing_time,
                'user_session': user_session
            }
            
            self.recent_predictions.append(prediction_data)
            self.prediction_counts[predicted_class] += 1
            self.confidence_scores.append(confidence)
            self.processing_times.append(processing_time)
            
            # Update metrics
            self.total_predictions += 1
            if confidence > 0.5:  # Consider predictions with >50% confidence as successful
                self.successful_predictions += 1
            
            self.average_confidence = np.mean(list(self.confidence_scores)) if self.confidence_scores else 0.0
            
            # Update hourly stats
            current_hour = datetime.now().hour
            self.hourly_stats[current_hour] += 1
            
            logger.info(f"📊 Recorded prediction: {predicted_class} ({confidence:.2%})")
            
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
    
    def get_realtime_stats(self):
        """Get current real-time statistics"""
        return {
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'success_rate': (self.successful_predictions / max(self.total_predictions, 1)) * 100,
            'average_confidence': self.average_confidence * 100,
            'active_users': len(self.active_users),
            'avg_processing_time': np.mean(list(self.processing_times)) if self.processing_times else 0.0,
            'predictions_last_hour': sum(list(self.hourly_stats.values())[-1:]),
            'most_common_disease': max(self.prediction_counts.items(), key=lambda x: x[1])[0] if self.prediction_counts else 'None'
        }
    
    def get_prediction_timeline(self, hours=24):
        """Get prediction timeline for the last N hours"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get predictions from last N hours
            since_time = datetime.now() - timedelta(hours=hours)
            cursor.execute('''
                SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                FROM predictions 
                WHERE timestamp >= ?
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            ''', (since_time,))
            
            results = cursor.fetchall()
            conn.close()
            
            # Create 24-hour timeline
            timeline = {str(i).zfill(2): 0 for i in range(24)}
            for hour, count in results:
                timeline[hour] = count
            
            return {
                'labels': list(timeline.keys()),
                'data': list(timeline.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction timeline: {e}")
            return {'labels': [], 'data': []}
    
    def get_disease_distribution(self):
        """Get current disease prediction distribution"""
        if not self.prediction_counts:
            return {'labels': [], 'data': []}
        
        labels = list(self.prediction_counts.keys())
        data = list(self.prediction_counts.values())
        
        return {
            'labels': labels,
            'data': data,
            'colors': [
                '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
                '#9966FF', '#FF9F40', '#FF6384'
            ][:len(labels)]
        }
    
    def get_confidence_trend(self):
        """Get confidence score trend"""
        if not self.confidence_scores:
            return {'labels': [], 'data': []}
        
        # Get last 20 confidence scores with timestamps
        scores = list(self.confidence_scores)[-20:]
        labels = [f"Pred {i+1}" for i in range(len(scores))]
        data = [score * 100 for score in scores]  # Convert to percentage
        
        return {
            'labels': labels,
            'data': data
        }
    
    def get_performance_metrics(self):
        """Get detailed performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get today's stats
            today = datetime.now().date()
            cursor.execute('''
                SELECT COUNT(*) as total, AVG(confidence) as avg_conf, AVG(processing_time) as avg_time
                FROM predictions 
                WHERE DATE(timestamp) = ?
            ''', (today,))
            
            today_stats = cursor.fetchone()
            
            # Get weekly trend
            week_ago = datetime.now() - timedelta(days=7)
            cursor.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM predictions 
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (week_ago,))
            
            weekly_data = cursor.fetchall()
            conn.close()
            
            return {
                'today': {
                    'total_predictions': today_stats[0] or 0,
                    'avg_confidence': (today_stats[1] or 0) * 100,
                    'avg_processing_time': today_stats[2] or 0
                },
                'weekly_trend': {
                    'labels': [row[0] for row in weekly_data],
                    'data': [row[1] for row in weekly_data]
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {'today': {}, 'weekly_trend': {'labels': [], 'data': []}}
    
    def add_active_user(self, user_session):
        """Add an active user session"""
        self.active_users.add(user_session)
    
    def remove_active_user(self, user_session):
        """Remove an active user session"""
        self.active_users.discard(user_session)
    
    def start_background_tasks(self):
        """Start background tasks for data cleanup and aggregation"""
        def cleanup_old_data():
            while True:
                try:
                    # Clean up old predictions (keep last 30 days)
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    thirty_days_ago = datetime.now() - timedelta(days=30)
                    cursor.execute('DELETE FROM predictions WHERE timestamp < ?', (thirty_days_ago,))
                    
                    conn.commit()
                    conn.close()
                    
                    logger.info("🧹 Cleaned up old analytics data")
                    
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
                
                # Sleep for 24 hours
                time.sleep(24 * 60 * 60)
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=cleanup_old_data, daemon=True)
        cleanup_thread.start()

# Global analytics instance
analytics = RealTimeAnalytics()

def create_realtime_app(main_app):
    """Create real-time analytics routes for the main Flask app"""
    
    @main_app.route('/analytics/dashboard')
    def analytics_dashboard():
        """Live analytics dashboard page with real-time WebSocket support"""
        return render_template('analytics_dashboard.html')
    
    @main_app.route('/api/analytics/stats')
    def get_analytics_stats():
        """Get real-time analytics statistics"""
        return jsonify(analytics.get_realtime_stats())
    
    @main_app.route('/api/analytics/timeline')
    def get_prediction_timeline():
        """Get prediction timeline"""
        hours = request.args.get('hours', 24, type=int)
        return jsonify(analytics.get_prediction_timeline(hours))
    
    @main_app.route('/api/analytics/distribution')
    def get_disease_distribution():
        """Get disease prediction distribution"""
        return jsonify(analytics.get_disease_distribution())
    
    @main_app.route('/api/analytics/confidence')
    def get_confidence_trend():
        """Get confidence score trend"""
        return jsonify(analytics.get_confidence_trend())
    
    @main_app.route('/api/analytics/performance')
    def get_performance_metrics():
        """Get performance metrics"""
        return jsonify(analytics.get_performance_metrics())
    
    return analytics

# Utility functions for integration
def record_prediction_analytics(predicted_class, confidence, processing_time, user_session, image_size=None):
    """Utility function to record prediction analytics"""
    analytics.record_prediction(predicted_class, confidence, processing_time, user_session, image_size)

def add_active_user_analytics(user_session):
    """Utility function to add active user"""
    analytics.add_active_user(user_session)

def remove_active_user_analytics(user_session):
    """Utility function to remove active user"""
    analytics.remove_active_user(user_session)

# Create global analytics instance
analytics = None

def create_realtime_app():
    """Create and configure the real-time analytics system"""
    global analytics
    analytics = RealTimeAnalytics()
    return analytics

class RealTimeAnalytics:
    def __init__(self, db_path='analytics.db'):
        self.db_path = db_path
        self.active_users = set()
        
    def add_active_user(self, user_session):
        """Add active user"""
        self.active_users.add(user_session)
        
    def remove_active_user(self, user_session):
        """Remove active user"""
        self.active_users.discard(user_session)
        
    def record_prediction(self, predicted_class, confidence, processing_time, user_session, image_size=None):
        """Record prediction analytics"""
        pass

class RealTimeAnalytics:
    def __init__(self, db_path='analytics.db'):
        self.db_path = db_path
        self.init_database()
        
        # In-memory storage for real-time data
        self.recent_predictions = deque(maxlen=100)  # Last 100 predictions
        self.active_users = set()
        self.prediction_counts = defaultdict(int)
        self.confidence_scores = deque(maxlen=50)
        self.hourly_stats = defaultdict(int)
        
        # Performance metrics
        self.total_predictions = 0
        self.successful_predictions = 0
        self.average_confidence = 0.0
        self.processing_times = deque(maxlen=20)
        
        # Start background tasks
        self.start_background_tasks()
    
    def init_database(self):
        """Initialize SQLite database for persistent analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                predicted_class TEXT,
                confidence REAL,
                processing_time REAL,
                user_session TEXT,
                image_size TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date DATE PRIMARY KEY,
                total_predictions INTEGER DEFAULT 0,
                unique_users INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.0,
                most_common_disease TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_usage (
                hour INTEGER PRIMARY KEY,
                prediction_count INTEGER DEFAULT 0,
                avg_processing_time REAL DEFAULT 0.0
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Analytics database initialized")
    
    def record_prediction(self, predicted_class, confidence, processing_time, user_session, image_size=None):
        """Record a new prediction for analytics"""
        try:
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (predicted_class, confidence, processing_time, user_session, image_size)
                VALUES (?, ?, ?, ?, ?)
            ''', (predicted_class, confidence, processing_time, user_session, image_size))
            conn.commit()
            conn.close()
            
            # Update in-memory data
            prediction_data = {
                'timestamp': datetime.now().isoformat(),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'processing_time': processing_time,
                'user_session': user_session
            }
            
            self.recent_predictions.append(prediction_data)
            self.prediction_counts[predicted_class] += 1
            self.confidence_scores.append(confidence)
            self.processing_times.append(processing_time)
            
            # Update metrics
            self.total_predictions += 1
            if confidence > 0.5:  # Consider predictions with >50% confidence as successful
                self.successful_predictions += 1
            
            self.average_confidence = np.mean(list(self.confidence_scores)) if self.confidence_scores else 0.0
            
            # Update hourly stats
            current_hour = datetime.now().hour
            self.hourly_stats[current_hour] += 1
            
            logger.info(f"📊 Recorded prediction: {predicted_class} ({confidence:.2%})")
            
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
    
    def get_realtime_stats(self):
        """Get current real-time statistics"""
        return {
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'success_rate': (self.successful_predictions / max(self.total_predictions, 1)) * 100,
            'average_confidence': self.average_confidence * 100,
            'active_users': len(self.active_users),
            'avg_processing_time': np.mean(list(self.processing_times)) if self.processing_times else 0.0,
            'predictions_last_hour': sum(list(self.hourly_stats.values())[-1:]),
            'most_common_disease': max(self.prediction_counts.items(), key=lambda x: x[1])[0] if self.prediction_counts else 'None'
        }
    
    def get_prediction_timeline(self, hours=24):
        """Get prediction timeline for the last N hours"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get predictions from last N hours
            since_time = datetime.now() - timedelta(hours=hours)
            cursor.execute('''
                SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                FROM predictions 
                WHERE timestamp >= ?
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            ''', (since_time,))
            
            results = cursor.fetchall()
            conn.close()
            
            # Create 24-hour timeline
            timeline = {str(i).zfill(2): 0 for i in range(24)}
            for hour, count in results:
                timeline[hour] = count
            
            return {
                'labels': list(timeline.keys()),
                'data': list(timeline.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction timeline: {e}")
            return {'labels': [], 'data': []}
    
    def get_disease_distribution(self):
        """Get current disease prediction distribution"""
        if not self.prediction_counts:
            return {'labels': [], 'data': []}
        
        labels = list(self.prediction_counts.keys())
        data = list(self.prediction_counts.values())
        
        return {
            'labels': labels,
            'data': data,
            'colors': [
                '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
                '#9966FF', '#FF9F40', '#FF6384'
            ][:len(labels)]
        }
    
    def get_confidence_trend(self):
        """Get confidence score trend"""
        if not self.confidence_scores:
            return {'labels': [], 'data': []}
        
        # Get last 20 confidence scores with timestamps
        scores = list(self.confidence_scores)[-20:]
        labels = [f"Pred {i+1}" for i in range(len(scores))]
        data = [score * 100 for score in scores]  # Convert to percentage
        
        return {
            'labels': labels,
            'data': data
        }
    
    def get_performance_metrics(self):
        """Get detailed performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get today's stats
            today = datetime.now().date()
            cursor.execute('''
                SELECT COUNT(*) as total, AVG(confidence) as avg_conf, AVG(processing_time) as avg_time
                FROM predictions 
                WHERE DATE(timestamp) = ?
            ''', (today,))
            
            today_stats = cursor.fetchone()
            
            # Get weekly trend
            week_ago = datetime.now() - timedelta(days=7)
            cursor.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM predictions 
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (week_ago,))
            
            weekly_data = cursor.fetchall()
            conn.close()
            
            return {
                'today': {
                    'total_predictions': today_stats[0] or 0,
                    'avg_confidence': (today_stats[1] or 0) * 100,
                    'avg_processing_time': today_stats[2] or 0
                },
                'weekly_trend': {
                    'labels': [row[0] for row in weekly_data],
                    'data': [row[1] for row in weekly_data]
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {'today': {}, 'weekly_trend': {'labels': [], 'data': []}}
    
    def add_active_user(self, user_session):
        """Add an active user session"""
        self.active_users.add(user_session)
    
    def remove_active_user(self, user_session):
        """Remove an active user session"""
        self.active_users.discard(user_session)
    
    def start_background_tasks(self):
        """Start background tasks for data cleanup and aggregation"""
        def cleanup_old_data():
            while True:
                try:
                    # Clean up old predictions (keep last 30 days)
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    thirty_days_ago = datetime.now() - timedelta(days=30)
                    cursor.execute('DELETE FROM predictions WHERE timestamp < ?', (thirty_days_ago,))
                    
                    conn.commit()
                    conn.close()
                    
                    logger.info("🧹 Cleaned up old analytics data")
                    
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
                
                # Sleep for 24 hours
                time.sleep(24 * 60 * 60)
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=cleanup_old_data, daemon=True)
        cleanup_thread.start()

# Global analytics instance
analytics = RealTimeAnalytics()

def create_realtime_app(main_app):
    """Create real-time analytics routes for the main Flask app"""
    
    @main_app.route('/analytics/dashboard')
    def analytics_dashboard():
        """Live analytics dashboard page with real-time WebSocket support"""
        return render_template('analytics_dashboard.html')
    
    @main_app.route('/api/analytics/stats')
    def get_analytics_stats():
        """Get real-time analytics statistics"""
        return jsonify(analytics.get_realtime_stats())
    
    @main_app.route('/api/analytics/timeline')
    def get_prediction_timeline():
        """Get prediction timeline"""
        hours = request.args.get('hours', 24, type=int)
        return jsonify(analytics.get_prediction_timeline(hours))
    
    @main_app.route('/api/analytics/distribution')
    def get_disease_distribution():
        """Get disease prediction distribution"""
        return jsonify(analytics.get_disease_distribution())
    
    @main_app.route('/api/analytics/confidence')
    def get_confidence_trend():
        """Get confidence score trend"""
        return jsonify(analytics.get_confidence_trend())
    
    @main_app.route('/api/analytics/performance')
    def get_performance_metrics():
        """Get performance metrics"""
        return jsonify(analytics.get_performance_metrics())
    
    return analytics

# Utility functions for integration
def record_prediction_analytics(predicted_class, confidence, processing_time, user_session, image_size=None):
    """Utility function to record prediction analytics"""
    analytics.record_prediction(predicted_class, confidence, processing_time, user_session, image_size)

def add_active_user_analytics(user_session):
    """Utility function to add active user"""
    analytics.add_active_user(user_session)

def remove_active_user_analytics(user_session):
    """Utility function to remove active user"""
    analytics.remove_active_user(user_session)