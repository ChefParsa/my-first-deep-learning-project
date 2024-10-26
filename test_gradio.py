import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from train import Net

# Load your model once to avoid reloading every time
model = torch.load('best_model.pth', map_location=torch.device("cuda"))
# model.eval()  # Set model to evaluation mode

def preprocess_image(image):
    # Convert NumPy array to PIL Image, then convert to grayscale
    image = Image.fromarray(image).convert('L')
    # Define the necessary transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    image = image.to("cuda")  # Move to GPU if available
    
    return image

def predict_class(image):
    image = preprocess_image(image)
    with torch.no_grad():
        prediction = model(image)
        predict_class = (torch.argmax(prediction, dim=1) + 1).item()
    
    return str(predict_class)

if __name__ == '__main__':
    demo = gr.Interface(
        fn=predict_class,
        inputs=gr.Image(type="numpy", image_mode='L'),  # Expect image as NumPy array in grayscale (L mode)
        outputs="text",
        title="Image Classifier",
        description="Upload an image and the model will classify it."
    )

    demo.launch()
