"""
Flask application for the Fantasy League Optimization website.

This module serves the API endpoints and static files for the website.
"""

from flask import Flask, send_from_directory, request, jsonify
import os
import sys
from flask_cors import CORS

# Add project root to path
sys.path.append('/home/ubuntu/fantasy_league_dashboard')

# Import API module
from api import app as api_blueprint

# Create Flask app
app = Flask(__name__, static_folder='interactive_website/dist')
CORS(app)  # Enable CORS for all routes

# Register API blueprint
app.register_blueprint(api_blueprint, url_prefix='/api')

# Direct optimization endpoints (not under /api prefix)
@app.route('/optimize/start', methods=['POST'])
def optimize_start():
    """Endpoint to start optimization."""
    try:
        # Get request data
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
            
        # Forward to API endpoint
        from api import start_optimization
        return start_optimization()
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error starting optimization: {str(e)}'
        }), 500

@app.route('/optimize/status', methods=['GET'])
def optimize_status():
    """Endpoint to get optimization status."""
    try:
        # Forward to API endpoint
        from api import get_optimization_status
        return get_optimization_status()
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting optimization status: {str(e)}'
        }), 500

@app.route('/optimize/events', methods=['GET'])
def optimize_events():
    """Endpoint to stream optimization events."""
    try:
        # Forward to API endpoint
        from api import get_optimization_events
        return get_optimization_events()
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting optimization events: {str(e)}'
        }), 500

@app.route('/optimize/stop', methods=['POST'])
def optimize_stop():
    """Endpoint to stop optimization."""
    try:
        # Forward to API endpoint
        from api import stop_optimization
        return stop_optimization()
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error stopping optimization: {str(e)}'
        }), 500

@app.route('/optimize/reset', methods=['POST'])
def optimize_reset():
    """Endpoint to reset optimization state."""
    try:
        # Forward to API endpoint
        from api import reset_optimization
        return reset_optimization()
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error resetting optimization: {str(e)}'
        }), 500

# Serve static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Create data directory
os.makedirs('/home/ubuntu/fantasy_league_dashboard/data', exist_ok=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
