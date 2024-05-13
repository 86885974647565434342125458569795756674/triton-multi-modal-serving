import http.server
import queue
import time
import numpy as np
import json
import multiprocessing
from blip_vqa_process import change_batch_size,blip_vqa_process_queue

def http_server(request_queue,request_events,processed_results):
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
            # print(post_data)
            # Extract data from the received bytes
            images = post_data.get('images')
            texts = post_data.get('texts')

            images=np.array([image.encode('utf-8') for image in images])
            texts=np.array([text.encode('utf-8') for text in texts])

            # Put the request in the queue
            request_queue.put((request_id,images,texts))

            request_events[request_id]=0

            while request_events[request_id]==0:
                pass

            del request_events[request_id]

            result = processed_results.get(request_id, b"Post data not available")

            result=[text.decode('utf-8') for text in result]

            result_str = "\n".join(result)  # Convert the list to a string with each element on a new line
            result_bytes = result_str.encode('utf-8') 
            # Send the result back to the client
            self._set_headers()
            self.wfile.write(result_bytes)


    with http.server.HTTPServer(("", 8971), RequestHandler) as httpd:
        print("Server started at port 8971")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()


if __name__ == "__main__":
    try:
        batch_size_queue=multiprocessing.Queue()
        # queue is not the fastest way, maybe pipe
        time_interval=1
        
        request_queue=multiprocessing.Queue()
        manager = multiprocessing.Manager()
        request_events=manager.dict()
        processed_results = manager.dict()

        # Create a list to hold process objects
        processes = []

        change_batch_size_process = multiprocessing.Process(target=change_batch_size, args=(batch_size_queue,time_interval))
        processes.append(change_batch_size_process)
        change_batch_size_process.start()

        blip_vqa_process_queue_process = multiprocessing.Process(target=blip_vqa_process_queue, args=(request_queue,request_events,processed_results,batch_size_queue,))
        processes.append(blip_vqa_process_queue_process)
        blip_vqa_process_queue_process.start()
        
        http_server_process = multiprocessing.Process(target=http_server, args=(request_queue,request_events,processed_results,))
        processes.append(http_server_process)
        http_server_process.start()
        
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.join()

