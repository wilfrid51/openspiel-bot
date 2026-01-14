#!/usr/bin/env python3
"""
Simple HTTP server to serve Hex game state updates for real-time visualization
"""
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os

# Global state storage
_game_state = {
    'board': None,
    'size': 11,
    'moves': [],
    'next_move': None,
    'next_to_move': None,
    'last_update': 0
}

_lock = threading.Lock()

def update_game_state(board_state, size, moves, next_move=None, next_to_move=None):
    """Update the global game state"""
    global _game_state
    with _lock:
        _game_state = {
            'board': board_state,
            'size': size,
            'moves': moves,
            'next_move': next_move,
            'next_to_move': next_to_move,
            'last_update': _game_state['last_update'] + 1
        }

def get_game_state():
    """Get current game state"""
    with _lock:
        return _game_state.copy()

class HexStateHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/state':
            # Return current game state as JSON
            state = get_game_state()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(state).encode())
            
        elif parsed_path.path == '/hex_realtime.html' or parsed_path.path == '/realtime':
            # Serve the real-time HTML file
            try:
                html_path = os.path.join(os.path.dirname(__file__), 'hex_realtime.html')
                with open(html_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(f"Error: {e}".encode())
            
        elif parsed_path.path == '/hex.html' or parsed_path.path == '/':
            # Serve the HTML file
            try:
                html_path = os.path.join(os.path.dirname(__file__), 'hex.html')
                with open(html_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(f"Error: {e}".encode())
                
        elif parsed_path.path.endswith('.css'):
            # Serve CSS files
            try:
                css_path = os.path.join(os.path.dirname(__file__), parsed_path.path.lstrip('/'))
                with open(css_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/css')
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(404)
                self.end_headers()
                
        elif parsed_path.path.endswith('.js'):
            # Serve JS files
            try:
                js_path = os.path.join(os.path.dirname(__file__), parsed_path.path.lstrip('/'))
                with open(js_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/javascript')
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests for state updates"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/update':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))

                # Update global state
                update_game_state(
                    board_state=data.get('board'),
                    size=data.get('size', 11),
                    moves=data.get('moves', []),
                    next_move=data.get('next_move'),
                    next_to_move=data.get('next_to_move')
                )
                
                # Send response - handle broken pipe gracefully
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'ok'}).encode())
                    self.wfile.flush()
                except (BrokenPipeError, OSError):
                    # Client closed connection early - this is fine, state was updated
                    pass
            except Exception as e:
                # Try to send error response, but don't crash if connection is broken
                try:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e)}).encode())
                except (BrokenPipeError, OSError):
                    pass
        else:
            try:
                self.send_response(404)
                self.end_headers()
            except (BrokenPipeError, OSError):
                pass
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

def start_server(port=8001):
    """Start the HTTP server"""
    server = HTTPServer(('0.0.0.0', port), HexStateHandler)  # 0.0.0.0 to allow external access
    print(f"Hex state server running on http://localhost:{port}")
    print(f"Open http://localhost:{port}/hex_realtime.html for real-time visualization")
    print(f"Or http://localhost:{port}/hex.html for static board")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    start_server(port)
