import requests
import base64

# Path to your image
image_path = '../QR_d_best/1_1_B_8.jpg'

# URL of the API server
url = 'http://127.0.0.1:8000/predict'  # Make sure the endpoint is correct

# Read the image and encode it in base64
with open(image_path, 'rb') as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

# Prepare the payload
payload = {
    "image_data": image_data
}

# Send the POST request
response = requests.post(url, json=payload)

# Check the response
if response.status_code == 200:
    print(response.json())  # This will print the prediction result
else:
    print(f"Failed to get prediction. Status code: {response.status_code}")
