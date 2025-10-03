<<<<<<< HEAD

"""
WebSocket-based real-time analytics for Plant Disease Analysis
"""

from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import threading
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class WebSocketAnalytics:
    def __init__(self, app, analytics_instance):
        self.socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
        self.analytics = analytics_instance
        self.connected_clients = set()
        
        # Set up WebSocket event handlers
        self.setup_handlers()
        
        # Start background broadcasting
        self.start_broadcasting()
    
    def setup_handlers(self):
        """Set up WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            client_id = request.sid
            self.connected_clients.add(client_id)
            join_room('analytics')
            
            logger.info(f"📡 Client connected: {client_id}")
            
            # Send initial data to the new client
            emit('initial_data', {
                'stats': self.analytics.get_realtime_stats(),
                'timeline': self.analytics.get_prediction_timeline(),
                'distribution': self.analytics.get_disease_distribution(),
                'confidence': self.analytics.get_confidence_trend()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            client_id = request.sid
            self.connected_clients.discard(client_id)
            leave_room('analytics')
            
            logger.info(f"📡 Client disconnected: {client_id}")
        
        @self.socketio.on('request_update')
        def handle_update_request(data):
            """Handle manual update requests"""
            chart_type = data.get('chart_type', 'all')
            
            if chart_type == 'stats' or chart_type == 'all':
                emit('stats_update', self.analytics.get_realtime_stats())
            
            if chart_type == 'timeline' or chart_type == 'all':
                emit('timeline_update', self.analytics.get_prediction_timeline())
            
            if chart_type == 'distribution' or chart_type == 'all':
                emit('distribution_update', self.analytics.get_disease_distribution())
            
            if chart_type == 'confidence' or chart_type == 'all':
                emit('confidence_update', self.analytics.get_confidence_trend())
    
    def broadcast_prediction_update(self, prediction_data):
        """Broadcast new prediction to all connected clients"""
        if self.connected_clients:
            self.socketio.emit('new_prediction', {
                'prediction': prediction_data,
                'timestamp': datetime.now().isoformat(),
                'stats': self.analytics.get_realtime_stats()
            }, room='analytics')
            
            logger.info(f"📡 Broadcasted prediction update to {len(self.connected_clients)} clients")
    
    def start_broadcasting(self):
        """Start background broadcasting of updates"""
        def broadcast_updates():
            while True:
                try:
                    if self.connected_clients:
                        # Broadcast updated stats every 30 seconds
                        self.socketio.emit('stats_update', 
                                         self.analytics.get_realtime_stats(), 
                                         room='analytics')
                        
                        # Broadcast chart updates every 60 seconds
                        self.socketio.emit('charts_update', {
                            'timeline': self.analytics.get_prediction_timeline(),
                            'distribution': self.analytics.get_disease_distribution(),
                            'confidence': self.analytics.get_confidence_trend()
                        }, room='analytics')
                        
                        logger.info(f"📡 Broadcasted updates to {len(self.connected_clients)} clients")
                    
                    time.sleep(30)  # Broadcast every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in broadcast thread: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start broadcasting thread
        broadcast_thread = threading.Thread(target=broadcast_updates, daemon=True)
        broadcast_thread.start()
        logger.info("📡 WebSocket broadcasting started")
    
    def get_socketio(self):
        """Get the SocketIO instance"""
        return self.socketio

def create_websocket_analytics(app, analytics_instance):
    """Create WebSocket analytics instance"""
=======
"""
WebSocket-based real-time analytics for Plant Disease Analysis
"""

from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import threading
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class WebSocketAnalytics:
    def __init__(self, app, analytics_instance):
        self.socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
        self.analytics = analytics_instance
        self.connected_clients = set()
        
        # Set up WebSocket event handlers
        self.setup_handlers()
        
        # Start background broadcasting
        self.start_broadcasting()
    
    def setup_handlers(self):
        """Set up WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            client_id = request.sid
            self.connected_clients.add(client_id)
            join_room('analytics')
            
            logger.info(f"📡 Client connected: {client_id}")
            
            # Send initial data to the new client
            emit('initial_data', {
                'stats': self.analytics.get_realtime_stats(),
                'timeline': self.analytics.get_prediction_timeline(),
                'distribution': self.analytics.get_disease_distribution(),
                'confidence': self.analytics.get_confidence_trend()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            client_id = request.sid
            self.connected_clients.discard(client_id)
            leave_room('analytics')
            
            logger.info(f"📡 Client disconnected: {client_id}")
        
        @self.socketio.on('request_update')
        def handle_update_request(data):
            """Handle manual update requests"""
            chart_type = data.get('chart_type', 'all')
            
            if chart_type == 'stats' or chart_type == 'all':
                emit('stats_update', self.analytics.get_realtime_stats())
            
            if chart_type == 'timeline' or chart_type == 'all':
                emit('timeline_update', self.analytics.get_prediction_timeline())
            
            if chart_type == 'distribution' or chart_type == 'all':
                emit('distribution_update', self.analytics.get_disease_distribution())
            
            if chart_type == 'confidence' or chart_type == 'all':
                emit('confidence_update', self.analytics.get_confidence_trend())
    
    def broadcast_prediction_update(self, prediction_data):
        """Broadcast new prediction to all connected clients"""
        if self.connected_clients:
            self.socketio.emit('new_prediction', {
                'prediction': prediction_data,
                'timestamp': datetime.now().isoformat(),
                'stats': self.analytics.get_realtime_stats()
            }, room='analytics')
            
            logger.info(f"📡 Broadcasted prediction update to {len(self.connected_clients)} clients")
    
    def start_broadcasting(self):
        """Start background broadcasting of updates"""
        def broadcast_updates():
            while True:
                try:
                    if self.connected_clients:
                        # Broadcast updated stats every 30 seconds
                        self.socketio.emit('stats_update', 
                                         self.analytics.get_realtime_stats(), 
                                         room='analytics')
                        
                        # Broadcast chart updates every 60 seconds
                        self.socketio.emit('charts_update', {
                            'timeline': self.analytics.get_prediction_timeline(),
                            'distribution': self.analytics.get_disease_distribution(),
                            'confidence': self.analytics.get_confidence_trend()
                        }, room='analytics')
                        
                        logger.info(f"📡 Broadcasted updates to {len(self.connected_clients)} clients")
                    
                    time.sleep(30)  # Broadcast every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in broadcast thread: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start broadcasting thread
        broadcast_thread = threading.Thread(target=broadcast_updates, daemon=True)
        broadcast_thread.start()
        logger.info("📡 WebSocket broadcasting started")
    
    def get_socketio(self):
        """Get the SocketIO instance"""
        return self.socketio

def create_websocket_analytics(app, analytics_instance):
    """Create WebSocket analytics instance"""
>>>>>>> e1fcd1d8ea3d427a90f7cd895c6c465448981fcb
    return WebSocketAnalytics(app, analytics_instance)