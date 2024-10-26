import torch, torchvision, PIL, io, base64, os
from concurrent.futures import ThreadPoolExecutor
import litserve as ls
from train_ResNet50_custom import Net
from torchvision.transforms import transforms, Compose

precision = torch.bfloat16

img_transform = Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.61], std=[0.14])
])


class ImageClassifierAPI(ls.LitAPI):
    def setup(self, device):
        self.image_processing = img_transform
        self.model = Net.load_from_checkpoint("./checkpoint/ResNet50 on QR_d_best-epoch=09-val_loss=0.01.ckpt").to(device).to(precision)
        self.pool = ThreadPoolExecutor(os.cpu_count())
        
    def decode_request(self, request):
        image_data = request["image_data"]
        return image_data
    
    def batch(self, inputs):
        print(len(inputs))
        def process_batch(image_data):
            image = base64.b64decode(image_data)
            pil_image = PIL.Image.open(io.BytesIO(image)).convert("L")
            return self.image_processing(pil_image)

        batched_inputs = list(self.pool.map(process_batch, inputs))
        return torch.stack(batched_inputs).to(self.device).to(precision)

    def predict(self, x):
        with torch.inference_mode():
            outputs = self.model(x)
            _, predictions = torch.max(outputs, 1)
        return predictions + 1
    
    def unbatch(self, output):
        return output.tolist()

    def encode_response(self, output):
        return {"output": output}

if __name__ == "__main__":
    api = ImageClassifierAPI()
    server = ls.LitServer(api, accelerator="gpu", devices=1, max_batch_size=16, workers_per_device=4, batch_timeout=0.01)
    server.run(port=8000)