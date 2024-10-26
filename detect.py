import torch
from torchvision import transforms
import argparse
from torchvision.io import read_image
from train import Net

# Define the function to load the model
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device("cuda"))
    model.eval()  # Set model to evaluation mode
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
    
    image = image.to("cuda")
    return image

# Define the function to make a prediction
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1)
        predicted_class = predicted_class + 1
    return predicted_class.item()

# Main function to handle the input arguments and run the prediction
def main():
    parser = argparse.ArgumentParser(description="Image Prediction Script")
    parser.add_argument('-img', '--image_path', type=str, required=True, help="Path to the image")
    parser.add_argument('-model', '--model_path', type=str, default='best_model.pth', help="Path to the model")

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
