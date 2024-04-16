import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
model = SamModel.from_pretrained("/Users/rstory/Repositories/SlimSAM-uniform-77")
processor = SamProcessor.from_pretrained("/Users/rstory/Repositories/SlimSAM-uniform-77")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]] # 2D localization of a window
inputs = processor(raw_image, input_points=input_points, return_tensors="pt")
outputs = model(**inputs)
masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
scores = outputs.iou_scores

# Convert the masks to PIL images
mask_images = [processor.image_processor.tensor_to_pil(mask) for mask in masks]
processor.image_processor.

# Display the masks
mask_images[0].show()