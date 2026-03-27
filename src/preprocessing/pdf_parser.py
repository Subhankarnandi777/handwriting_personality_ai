import fitz  # PyMuPDF
import cv2
import numpy as np

def extract_images_from_pdf(pdf_path: str):
    """
    Extracts all images from a PDF or rasterizes the pages if no embedded images exist.
    Returns a list of OpenCV BGR images.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Failed to open PDF: {e}")
        return []

    images = []
    
    for i in range(len(doc)):
        page = doc[i]
        image_list = page.get_images(full=True)
        
        # If the PDF contains embedded images, extract them directly
        if image_list:
            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert bytes to cv2 image
                np_arr = np.frombuffer(image_bytes, np.uint8)
                img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img_cv2 is not None:
                    images.append(img_cv2)
        else:
            # If no embedded images, rasterize the PDF page itself
            pix = page.get_pixmap(dpi=300)
            img_cv2 = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)
            if img_cv2 is not None:
                images.append(img_cv2)
                
    return images
