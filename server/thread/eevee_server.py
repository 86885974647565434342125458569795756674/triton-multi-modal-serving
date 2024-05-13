import http.server
import socketserver
import queue
import threading
import time
import numpy as np
from PIL import Image
import ast
import torch
import json
import random
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import tritonclient.http as httpclient
from tritonclient.utils import *
from start_blip_vqa_tasks import blip_vqa_visual_encoder_task,blip_vqa_text_encoder_task,blip_vqa_text_decoder_task

# Define a queue to store incoming POST requests
request_queue = queue.Queue()

# Define a dictionary to store processed results
processed_results = {}

# Define a dictionary to store threading Events for each request
request_events = {}

# Define a lock for accessing shared resources
processed_results_lock = threading.Lock()
request_events_lock = threading.Lock()

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

        # Create an event for this request
        request_event = threading.Event()
        with request_events_lock:
            request_events[request_id] = request_event

        # Put the request in the queue
        request_queue.put((request_id,images,texts))

        # Wait for the result to be available
        request_event.wait()

        # Get the result for this request
        with processed_results_lock:
            result = processed_results.get(request_id, b"Post data not available")

        result=[text.decode('utf-8') for text in result]

        result_str = "\n".join(result)  # Convert the list to a string with each element on a new line
        result_bytes = result_str.encode('utf-8') 
        # Send the result back to the client
        self._set_headers()
        self.wfile.write(result_bytes)

# Define a function to process the queue
def blip_vqa_process_queue(batch_size_queue):
    blip_vqa_visual_encoder_blip_vqa_text_encoder_queue=queue.Queue()
    blip_vqa_text_encoder_blip_vqa_text_decoder_queue=queue.Queue()

    blip_vqa_visual_encoder_batches_queue=queue.Queue()
    blip_vqa_visual_encoder_task_thread=threading.Thread(target=blip_vqa_visual_encoder_task,args=(blip_vqa_visual_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,))
    blip_vqa_visual_encoder_task_thread.start()

    blip_vqa_text_encoder_batches_queue=queue.Queue()
    blip_vqa_text_encoder_task_thread=threading.Thread(target=blip_vqa_text_encoder_task,args=(blip_vqa_text_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue,))
    blip_vqa_text_encoder_task_thread.start()

    blip_vqa_text_decoder_batches_queue=queue.Queue()
    blip_vqa_text_decoder_batches_return_queue=queue.Queue()
    blip_vqa_text_decoder_task_thread=threading.Thread(target=blip_vqa_text_decoder_task,args=(blip_vqa_text_decoder_batches_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue,blip_vqa_text_decoder_batches_return_queue,))
    blip_vqa_text_decoder_task_thread.start()

    blip_vqa_visual_encoder_batch_size,blip_vqa_text_encoder_batch_size,blip_vqa_text_decoder_batch_size=1,1,1
    
    while True:
        try:
            batch_sizes=batch_size_queue.get(block=False)
            if batch_sizes is None:
                break
            blip_vqa_visual_encoder_batch_size,blip_vqa_text_encoder_batch_size,blip_vqa_text_decoder_batch_size=batch_sizes[0],batch_sizes[1],batch_sizes[2]
        except queue.Empty:
            pass
        if not request_queue.empty():
            request_ids, batch_nums,images, texts=[],[],[],[]
            while not request_queue.empty():
                post_data = request_queue.get()
                request_id,image,text=post_data[0],post_data[1],post_data[2]
                request_ids.append(request_id)
                images.append(image)
                texts.append(text)
                batch_nums.append(image.shape[0])
            
            images=np.concatenate(images, axis=0)
            texts=np.concatenate(texts, axis=0)

            # cache remove image replication
             
            if images.shape[0]<=blip_vqa_visual_encoder_batch_size:
                blip_vqa_visual_encoder_batches=[images]
                num_blip_vqa_visual_encoder_batch=1
            else:
                num_blip_vqa_visual_encoder_batch=images.shape[0]//blip_vqa_visual_encoder_batch_size
                num_blip_vqa_visual_encoder_left=images.shape[0]-num_blip_vqa_visual_encoder_batch*blip_vqa_visual_encoder_batch_size
                if num_blip_vqa_visual_encoder_left!=0:
                    blip_vqa_visual_encoder_batches=np.split(images[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)+[images[-num_blip_vqa_visual_encoder_left:]]
                    num_blip_vqa_visual_encoder_batch+=1
                else:
                    blip_vqa_visual_encoder_batches=np.split(images,num_blip_vqa_visual_encoder_batch)

            if texts.shape[0]<=blip_vqa_text_encoder_batch_size:
                blip_vqa_text_encoder_batches=[texts]
                num_blip_vqa_text_encoder_batch=1
            else:
                num_blip_vqa_text_encoder_batch=texts.shape[0]//blip_vqa_text_encoder_batch_size
                num_blip_vqa_text_encoder_left=texts.shape[0]-num_blip_vqa_text_encoder_batch*blip_vqa_text_encoder_batch_size
                if num_blip_vqa_text_encoder_left!=0:
                    blip_vqa_text_encoder_batches=np.split(texts[:-num_blip_vqa_text_encoder_left],num_blip_vqa_text_encoder_batch)+[texts[-num_blip_vqa_text_encoder_left:]]
                    num_blip_vqa_text_encoder_batch+=1
                else:
                    blip_vqa_text_encoder_batches=np.split(texts,num_blip_vqa_text_encoder_batch)
            
            if texts.shape[0]<=blip_vqa_text_decoder_batch_size:
                blip_vqa_text_decoder_batches=[texts]
                num_blip_vqa_text_decoder_batch=1
            else:
                num_blip_vqa_text_decoder_batch=texts.shape[0]//blip_vqa_text_decoder_batch_size
                num_blip_vqa_text_decoder_left=texts.shape[0]-num_blip_vqa_text_decoder_batch*blip_vqa_text_decoder_batch_size
                if num_blip_vqa_text_decoder_left!=0:
                    blip_vqa_text_decoder_batches=np.split(texts[:-num_blip_vqa_text_decoder_left],num_blip_vqa_text_decoder_batch)+[texts[-num_blip_vqa_text_decoder_left:]]
                    num_blip_vqa_text_decoder_batch+=1
                else:
                    blip_vqa_text_decoder_batches=np.split(texts,num_blip_vqa_text_decoder_batch)

            blip_vqa_visual_encoder_batches_queue.put(blip_vqa_visual_encoder_batches,block=False)            
            blip_vqa_text_encoder_batches_queue.put(blip_vqa_text_encoder_batches,block=False)
            blip_vqa_text_decoder_batches_queue.put(blip_vqa_text_decoder_batches,block=False)

            batch_count=0
            now_left=None
            for _ in range(num_blip_vqa_text_decoder_batch):
                blip_vqa_text_decoder_batches_return=blip_vqa_text_decoder_batches_return_queue.get()
                if now_left is None:
                    now_left=blip_vqa_text_decoder_batches_return
                else:
                    now_left=np.concatenate([now_left,blip_vqa_text_decoder_batches_return],axis=0)
                while batch_count<len(batch_nums) and now_left.shape[0]>=batch_nums[batch_count]:
                    post_return=now_left[:batch_nums[batch_count]]
                    now_left=now_left[batch_nums[batch_count]:]
                    # print(post_return,batch_nums[batch_count],batch_count)
                    with processed_results_lock:
                        processed_results[request_ids[batch_count]] = post_return
                    with request_events_lock:
                        request_event = request_events.get(request_ids[batch_count])
                    request_event.set()
                    batch_count+=1
            
    
    blip_vqa_visual_encoder_batches_queue.put(None,block=False)
    blip_vqa_visual_encoder_task_thread.join()

    blip_vqa_text_encoder_batches_queue.put(None,block=False)
    blip_vqa_text_encoder_task_thread.join()

    blip_vqa_text_decoder_batches_queue.put(None,block=False)
    blip_vqa_text_decoder_task_thread.join()

def chang_batch_size(batch_size_queue):
    while True:
        try:
            # Attempt to access the first item in the queue
            first_item = batch_size_queue.queue[0]
            if first_item is None:
                break
        except IndexError:
            pass
        time.sleep(15)
        a,b,c=random.randint(1, 10),random.randint(1, 10),random.randint(1, 10)
        print(f"batch size:{a},{b},{c}")
        batch_size_queue.put((a,b,c),block=False)

if __name__=="__main__":
    batch_size_queue=queue.Queue()
    batch_size_queue.put((3,3,3),block=False)
    change_batch_size_thread = threading.Thread(target=chang_batch_size,args=(batch_size_queue,))
    change_batch_size_thread.start()
    blip_vqa_process_thread = threading.Thread(target=blip_vqa_process_queue,args=(batch_size_queue,))
    blip_vqa_process_thread.start()

    with socketserver.TCPServer(("", 8971), RequestHandler) as httpd:
        print("Server started at port 8971")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

    batch_size_queue.put(None,block=False)
    batch_size_queue.put(None,block=False)
    change_batch_size_thread.join()
    blip_vqa_process_thread.join()
