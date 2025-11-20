from inspect import cleandoc
import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SAM3Segmentation:
    """
    SAM3 Segmentation Node
    
    Performs image segmentation using SAM3 (Segment Anything Model 3) with text prompts.
    Takes an image and a text prompt, and outputs a visualization with segmentation masks,
    bounding boxes, and confidence scores.
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
    
    def load_model(self):
        """Lazy load the model on first use"""
        if self.model is None:
            print("Loading SAM3 model...")
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
            print("SAM3 model loaded successfully")
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to segment"}),
                "prompt": ("STRING", {
                    "multiline": False,
                    "default": "person",
                    "tooltip": "Text prompt describing what to segment (e.g., 'person', 'car', 'dog')"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum confidence score threshold (0.0 to 1.0)"
                }),
                "min_width_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "tooltip": "Minimum bounding box width in pixels"
                }),
                "min_height_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "tooltip": "Minimum bounding box height in pixels"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("segmented_image", "masks", "mask_combined")
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "segment"
    CATEGORY = "SAM3"
    
    def segment(self, image, prompt, threshold, min_width_pixels, min_height_pixels):
        """
        Perform segmentation on the input image using the text prompt
        
        Args:
            image: Input image tensor in ComfyUI format [B, H, W, C] with values in [0, 1]
            prompt: Text description of objects to segment
            threshold: Minimum confidence score threshold
            min_width_pixels: Minimum bounding box width in pixels
            min_height_pixels: Minimum bounding box height in pixels
            
        Returns:
            Tuple of (segmented_image, masks, mask_combined)
        """
        # Load model if not already loaded
        self.load_model()
        
        # Convert ComfyUI image format [B, H, W, C] to PIL Image
        # ComfyUI images are in range [0, 1], convert to [0, 255]
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # Set up inference state
        inference_state = self.processor.set_image(pil_image)
        
        # Run segmentation with text prompt
        output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
        
        # Get results
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        # Filter by threshold, minimum width, and minimum height
        filtered_indices = []
        for i, (score, box) in enumerate(zip(scores, boxes)):
            if score >= threshold:
                # Calculate box dimensions
                x1, y1, x2, y2 = box
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width >= min_width_pixels and box_height >= min_height_pixels:
                    filtered_indices.append(i)
        
        masks = [masks[i] for i in filtered_indices]
        boxes = [boxes[i] for i in filtered_indices]
        scores = [scores[i] for i in filtered_indices]
        
        print(f"SAM3 found {len(masks)} object(s) matching '{prompt}' with score >= {threshold}, width >= {min_width_pixels}px, height >= {min_height_pixels}px")
        
        # Create visualization
        result_img = pil_image.copy()
        
        # Process each detection
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            # Convert mask to numpy if it's a tensor
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            
            # Squeeze mask to remove extra dimensions
            mask = np.squeeze(mask)
            
            # Resize mask if needed to match image dimensions
            if mask.shape != (pil_image.size[1], pil_image.size[0]):
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img = mask_img.resize(pil_image.size, Image.NEAREST)
                mask = np.array(mask_img) > 0
            
            # Create colored overlay for mask
            mask_array = np.zeros((pil_image.size[1], pil_image.size[0], 4), dtype=np.uint8)
            
            # Apply mask with semi-transparent color
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            mask_color = colors[i % len(colors)]
            mask_array[mask] = (*mask_color, 100)  # Semi-transparent
            
            mask_overlay = Image.fromarray(mask_array, 'RGBA')
            result_img = Image.alpha_composite(result_img.convert('RGBA'), mask_overlay).convert('RGB')
            
            # Draw bounding box
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(result_img)
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
        
        # Convert result back to ComfyUI format [B, H, W, C] with values in [0, 1]
        result_np = np.array(result_img).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)
        
        # Prepare mask batch - ComfyUI masks are [B, H, W] with values in [0, 1]
        mask_list = []
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            # Convert mask to numpy if it's a tensor
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            
            # Squeeze mask to remove extra dimensions
            mask = np.squeeze(mask)
            
            # Resize mask if needed to match image dimensions
            if mask.shape != (pil_image.size[1], pil_image.size[0]):
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img = mask_img.resize(pil_image.size, Image.NEAREST)
                mask = (np.array(mask_img) / 255.0).astype(np.float32)
            else:
                mask = mask.astype(np.float32)
            
            mask_list.append(mask)
        
        # Stack masks into batch tensor [B, H, W]
        if len(mask_list) > 0:
            masks_tensor = torch.from_numpy(np.stack(mask_list, axis=0))
            # Create combined mask by taking maximum across all masks
            combined_mask = np.maximum.reduce(mask_list)
            combined_mask_tensor = torch.from_numpy(combined_mask).unsqueeze(0)
        else:
            # Return empty mask if no detections
            masks_tensor = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
            combined_mask_tensor = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
        
        return (result_tensor, masks_tensor, combined_mask_tensor)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SAM3Segmentation": SAM3Segmentation
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3Segmentation": "SAM3 Segmentation"
}
