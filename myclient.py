import requests
import threading
import numpy as np
import json

def send_post_request(images,texts):
    url = 'http://localhost:8971'  # Update with your server's address

    # Create a dictionary to hold the data
    data = {'images': images, 'texts': texts}
    data = json.dumps(data)

    response = requests.post(url, data=data)
    print(response.text)

if __name__ == '__main__':
    # Define the data to be sent in each POST request
    request_num=1
    batch_size=20
    post_data = [
        (["/workspace/demos/images/merlion.png"]*batch_size,["where is the woman sitting?"]*batch_size),
    ]*request_num

    
    # Send each POST request in a separate thread
    threads = []
    for data in post_data:
        thread = threading.Thread(target=send_post_request, args=(data[0],data[1],))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
