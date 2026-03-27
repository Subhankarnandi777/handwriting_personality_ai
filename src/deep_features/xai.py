import torch
import cv2
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from src.utils.config import DEEP

def generate_activation_heatmap(image_bgr: np.ndarray) -> np.ndarray:
    """
    Generates a pseudo-Grad-CAM activation heatmap from ResNet-50.
    Since the CNN is used as a feature extractor without a final classification
    layer trained end-to-end, we compute the spatial average activation of the
    last convolutional block to see what the network is "looking at".
    """
    try:
        device = torch.device(DEEP.get("device", "cpu"))
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.eval()
        model.to(device)
        
        # Hook into the last conv layer (layer4)
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
            
        model.layer4.register_forward_hook(get_activation('layer4'))
        
        # Prepare image
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            model(input_tensor)
            
        # Get activation map from layer 4
        act = activation['layer4'].squeeze() # shape [2048, 7, 7]
        
        # Average across channels and normalize
        heatmap = torch.mean(act, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        
        # Resize to original image size
        heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
        
        # Apply colormap
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Superimpose
        superimposed = cv2.addWeighted(image_bgr, 0.5, heatmap_color, 0.5, 0)
        return superimposed
        
    except Exception as e:
        print(f"Failed to generate XAI heatmap: {e}")
        return image_bgr # fallback
