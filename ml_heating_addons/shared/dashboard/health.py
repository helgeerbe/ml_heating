#!/usr/bin/env python3
"""
Simple health check endpoint for ML Heating Add-on
"""

import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'services': {
                    'dashboard': 'running',
                    'ml_system': self.check_ml_system(),
                    'config': self.check_config()
                }
            }
            
            self.wfile.write(json.dumps(health_status).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def check_ml_system(self):
        """Check ML system status"""
        if os.path.exists('/data/logs/ml_heating.log'):
            try:
                stat = os.stat('/data/logs/ml_heating.log')
                last_modified = datetime.fromtimestamp(stat.st_mtime)
                time_diff = datetime.now() - last_modified
                
                if time_diff.seconds < 300:  # 5 minutes
                    return 'active'
                else:
                    return 'inactive'
            except Exception:
                return 'error'
        return 'not_started'
    
    def check_config(self):
        """Check configuration status"""
        if os.path.exists('/data/options.json'):
            try:
                with open('/data/options.json', 'r') as f:
                    json.load(f)
                return 'valid'
            except Exception:
                return 'invalid'
        return 'missing'


def start_health_server():
    """Start simple health check server"""
    server = HTTPServer(('0.0.0.0', 3002), HealthHandler)
    print("Health check server starting on port 3002...")
    server.serve_forever()


if __name__ == "__main__":
    start_health_server()
