"""
============================================================================
 RESTAURANT ANALYTICS & ML - LOCAL WEB SERVER
============================================================================
 Run with:
     python server.py
 Open:
     http://localhost:8000
============================================================================
"""

import http.server
import os
import socketserver
import sys
import webbrowser

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, 'web')

class WebAppHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

def run_server():
    os.chdir(WEB_DIR)
    
    # Try port 8000, fallback to alternative if busy
    port = PORT
    for attempt in range(5):
        try:
            with socketserver.TCPServer(("", port), WebAppHandler) as httpd:
                url = f"http://localhost:{port}"
                print("\n" + "=" * 65)
                print("  RESTAURANT ANALYTICS & ML PREDICTOR WEB APPLICATION")
                print("=" * 65)
                print(f"  Server running locally at : {url}")
                print(f"  Web application root      : {WEB_DIR}")
                print("=" * 65)
                print("  Press Ctrl+C in terminal to stop the server.\n")
                
                try:
                    webbrowser.open(url)
                except Exception:
                    pass

                httpd.serve_forever()
                break
        except OSError:
            port += 1
            continue

if __name__ == '__main__':
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n[Server stopped]")
        sys.exit(0)
