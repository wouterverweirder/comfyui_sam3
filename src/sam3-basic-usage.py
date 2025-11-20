import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
# image = Image.open("sam3/assets/images/test_image.jpg")
# image = Image.open("C:\\ai\\ComfyUI\\comfyui-devine\\input\\PlotterInput_00117_.png")
# image = Image.open("C:\\ai\\ComfyUI\\comfyui-devine\\input\\PlotterInput_00124_.png")
# image = Image.open("C:\\ai\\ComfyUI\\comfyui-devine\\input\\PlotterInput_00237_.png")
# image = Image.open("C:\\ai\\ComfyUI\\comfyui-devine\\input\\PlotterInput_00247_.png")
image = Image.open("C:\\ai\\ComfyUI\\comfyui-devine\\input\\PlotterInput_00251_.png")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="people")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# Create visualization
import numpy as np
from PIL import ImageDraw, ImageFont

# Convert image to numpy array for processing
img_array = np.array(image)
result_img = image.copy()
draw = ImageDraw.Draw(result_img)

# Process each detection
for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
    # Convert mask to numpy if it's a tensor
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Squeeze mask to remove extra dimensions and ensure it matches image size
    mask = np.squeeze(mask)
    
    # Resize mask if needed to match image dimensions
    if mask.shape != (image.size[1], image.size[0]):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize(image.size, Image.NEAREST)
        mask = np.array(mask_img) > 0
    
    # Create colored overlay for mask
    mask_array = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
    
    # Apply mask with semi-transparent color
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    mask_color = color[i % len(color)]
    mask_array[mask] = (*mask_color, 100)  # Semi-transparent
    
    mask_overlay = Image.fromarray(mask_array, 'RGBA')
    result_img = Image.alpha_composite(result_img.convert('RGBA'), mask_overlay).convert('RGB')
    draw = ImageDraw.Draw(result_img)
    
    # Draw bounding box
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=mask_color, width=3)
    
    # Draw score text
    score_text = f"{score:.2f}"
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw text background
    text_bbox = draw.textbbox((x1, y1 - 25), score_text, font=font)
    draw.rectangle(text_bbox, fill=mask_color)
    draw.text((x1, y1 - 25), score_text, fill=(255, 255, 255), font=font)

# Save the result
result_img.save("result.jpg")
print(f"Result saved as result.jpg with {len(masks)} detections")

