import numpy as np
import cv2

def analyze_signature(binary_img, bboxes):
    """
    Extracts features specific to signatures.
    Signatures usually have:
    - High variation in stroke size
    - Underlines / flourishes
    - Different aspect ratios compared to normal text.
    """
    features = {}
    
    if len(bboxes) == 0:
        return {
            "sig_flourish": False, 
            "sig_underline": False, 
            "sig_aspect_ratio": 0.0
        }
        
    # Full bounding box around all components
    x_min = min(b[0] for b in bboxes)
    y_min = min(b[1] for b in bboxes)
    x_max = max(b[0] + b[2] for b in bboxes)
    y_max = max(b[1] + b[3] for b in bboxes)
    
    width = max(1, x_max - x_min)
    height = max(1, y_max - y_min)
    
    features["sig_aspect_ratio"] = width / height
    
    # Check for underline: a long horizontal component near the bottom
    has_underline = False
    for (x, y, w, h) in bboxes:
        if w > width * 0.6 and h < height * 0.2 and y > y_min + height * 0.7:
            has_underline = True
            break
            
    features["sig_underline"] = has_underline
    
    # Flourish: unusually large components that span most of the height but not width or vice-versa
    features["sig_flourish"] = any((h > height * 0.8 and w < width * 0.3) for (x, y, w, h) in bboxes)
    
    return features
