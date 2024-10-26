import torch
from torchvision import transforms
import argparse
from torchvision.io import read_image
import onnxruntime as ort
import numpy as np

# Define the function to load the model
def load_model(model_path):
    model = ort.InferenceSession(model_path)
    return model

# Define the function to preprocess the image
def preprocess_image(image_path):
    # Define transformations (adjust according to your model's requirements)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to match the input size expected by the model
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with ImageNet mean and std
    ])

    # Load image
    image = read_image(image_path)
    
    image = image.float() / 255.0
    
    # Apply transformations
    image = transform(image)
    
    # Add batch dimension (model expects batch size)
    image = image.unsqueeze(0)
    
    image = image.numpy()
    return image

# Define the function to make a prediction
def predict(model, image):
    input_name = model.get_inputs()[0].name
    ort_inputs = {input_name: image}
    ort_outs = model.run(None, ort_inputs)
    # converting list of np array to tensor is extremely slow so first convert the list to np array
    ort_outs = np.array(ort_outs)
    ort_outs = torch.Tensor(ort_outs)
    return (torch.argmax(ort_outs, dim=2) + 1).item()

# Main function to handle the input arguments and run the prediction
def main():
    # TODO Batch input
    parser = argparse.ArgumentParser(description="Image Prediction Script")
    parser.add_argument('-img', '--image_path', type=str, required=True, help="Path to the image")
    parser.add_argument('-model', '--model_path', type=str, default='model.onnx', help="Path to the model")

    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_path)

    # Preprocess the image
    image_tensor = preprocess_image(args.image_path)

    # Make prediction
    predicted_class = predict(model, image_tensor)

    # Output the result
    print(f'Predicted Class: {predicted_class}')

if __name__ == '__main__':
    main()
