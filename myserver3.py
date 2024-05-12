import http.server
import socketserver
import queue
import threading
import time
import numpy as np
import json


# Define a queue to store incoming POST requests
request_queue = queue.Queue()

# Define a dictionary to store processed results
processed_results = {}

# Define a dictionary to store threading Events for each request
request_events = {}

# Define a lock for accessing shared resources
processed_results_lock = threading.Lock()
request_events_lock = threading.Lock()

# Define a flag to signal when to stop processing the queue
stop_processing = False

# Define a class for handling HTTP requests
class RequestHandler(http.server.BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(b"<html><body><h1>GET request received!</h1></body></html>")

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        post_data = json.loads(post_data)
        request_id=time.time()
        print(post_data)
        # Extract data from the received bytes
        images = post_data.get('images')
        texts = post_data.get('texts')

        images=np.array([image.encode('utf-8') for image in images])
        texts=np.array([text.encode('utf-8') for text in texts])

        print(images)
        print(texts)
        self._set_headers()


with socketserver.TCPServer(("", 8000), RequestHandler) as httpd:
    print("Server started at port 8000")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()