
import onnxruntime
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
from torchvision import transforms

# --- Image Utility Functions ---
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# --- Preprocessing Constants ---
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# --- Preprocessing Transform ---
preprocess_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# --- Model Path (Assuming the model is in the same directory as the lambda) ---
MODEL_PATH = 'hair_classifier_v1.onnx'

# --- ONNX Inference Function ---
def predict_image(image_url):
    # Download and prepare image
    img = download_image(image_url)
    target_size = (200, 200) # Based on previous homework
    img_prepared = prepare_image(img, target_size)

    # Preprocess image
    img_tensor = preprocess_transform(img_prepared)
    input_data = np.expand_dims(img_tensor.numpy(), axis=0).astype(np.float32)

    # Create an ONNX Runtime session
    sess = onnxruntime.InferenceSession(MODEL_PATH)

    # Get input and output names (assuming 'input' and 'output' based on previous questions)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Run inference
    outputs = sess.run([output_name], {input_name: input_data})

    # Return the model output
    return outputs[0]

# --- Lambda Handler (for AWS Lambda or similar) ---
def lambda_handler(event, context):
    if 'url' not in event:
        return {
            'statusCode': 400,
            'body': 'Error: Missing image URL in event.'
        }

    image_url = event['url']
    try:
        prediction = predict_image(image_url)
        return {
            'statusCode': 200,
            'body': {'prediction': prediction.tolist()}}
    except Exception as e:
        return {
            'statusCode': 500,
            'body': f'Error during prediction: {str(e)}'
        }

