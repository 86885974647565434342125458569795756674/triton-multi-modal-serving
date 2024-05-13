import requests
import multiprocessing
import numpy as np
import json

def send_post_request(images,texts):
    url = 'http://localhost:8971'  # Update with your server's address

    # Create a dictionary to hold the data
    data = {'images': images, 'texts': texts}
    data = json.dumps(data)

    response = requests.post(url, data=data)
    print(response.text)
    print("finish")

if __name__ == '__main__':
    # Define the data to be sent in each POST request
    request_num=4
    batch_size=20
    post_data = [
        (["/workspace/demos/images/merlion.png"]*batch_size,["where is the woman sitting?"]*batch_size),
    ]*request_num

    
    # Send each POST request in a separate thread
    processes = []
    for data in post_data:
        process = multiprocessing.Process(target=send_post_request, args=(data[0],data[1],))
        process.start()
        processes.append(process)

    # Wait for all threads to complete
    for process in processes:
        process.join()
