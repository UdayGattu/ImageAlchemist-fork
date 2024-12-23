

# import cv2
# import numpy as np
# import torch
# from openai import OpenAI
# import os
# from dotenv import load_dotenv
# import logging
# import json

# # Initialize
# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# model = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# def detect_object(image):
#     """Detect product and ensure 85% frame fill"""
#     import torch.amp as amp
#     with amp.autocast(device_type='cpu'):
#         results = model(image[..., ::-1])
    
#     detections = results.xyxy[0].cpu().numpy()
#     if len(detections) == 0:
#         height, width, _ = image.shape
#         return (0, 0, width, height), image

#     largest_object = max(detections, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
#     x1, y1, x2, y2 = map(int, largest_object[:4])
    
#     # Ensure 85% frame fill
#     center_x = (x1 + x2) // 2
#     center_y = (y1 + y2) // 2
#     size = int(max(x2 - x1, y2 - y1) / 0.85)
    
#     x1 = max(0, center_x - size//2)
#     y1 = max(0, center_y - size//2)
#     x2 = min(image.shape[1], center_x + size//2)
#     y2 = min(image.shape[0], center_y + size//2)
    
#     return (x1, y1, x2, y2), image[y1:y2, x1:x2]

# def extract_features(image):
#     """Extract image quality metrics"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
#     return {
#         "brightness": np.mean(gray),
#         "contrast": np.std(gray),
#         "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var(),
#         "color_balance": {
#             "l": np.mean(lab[:,:,0]),
#             "a": np.mean(lab[:,:,1]),
#             "b": np.mean(lab[:,:,2])
#         }
#     }

# def query_llm(features):
#     """Get enhancement parameters from LLM"""
#     prompt = f"""
#     Analyze these product photo metrics and recommend optimal enhancement values:

#     Current Image Metrics:
#     - Brightness: {features['brightness']:.1f} (optimal range: 180-220)
#     - Contrast: {features['contrast']:.1f} (optimal range: 40-60)
#     - Sharpness: {features['sharpness']:.1f} (optimal range: 800-1200)
#     - Color Balance L: {features['color_balance']['l']:.1f}
#     - Color Balance a: {features['color_balance']['a']:.1f}
#     - Color Balance b: {features['color_balance']['b']:.1f}

#     Provide enhancement parameters as a Python dictionary:
#     {{
#         "brightness_factor": (float, e.g., 1.1),
#         "contrast_factor": (float, e.g., 1.2),
#         "sharpness_factor": (float, e.g., 0.2),
#         "saturation_factor": (float, e.g., 1.3)
#     }}

    
#     Base your values on these metrics:
#     Brightness: {features['brightness']:.1f} (target: 180-220)
#     Contrast: {features['contrast']:.1f} (target: 40-60)
#     Sharpness: {features['sharpness']:.1f} (target: 800-1200)
#     Color: L={features['color_balance']['l']:.1f}, a={features['color_balance']['a']:.1f}, b={features['color_balance']['b']:.1f}
    
#         Return ONLY a valid Python dictionary with these exact keys and values:
#     {{"brightness_factor": 1.2, "contrast_factor": 1.3, "sharpness_factor": 0.2, "saturation_factor": 1.15}}
#     """
    
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are an expert in product photo retouching,You are a JSON generator. Respond only with valid Python dictionary syntax."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.1
#         )
        
#         result = response.choices[0].message.content.strip()
#         recommendations = eval(result)
        
#         # Validate and clamp values
#         return {
#             "brightness_factor": max(0.9, min(1.3, recommendations["brightness_factor"])),
#             "contrast_factor": max(1.0, min(1.4, recommendations["contrast_factor"])),
#             "sharpness_factor": max(0.1, min(0.3, recommendations["sharpness_factor"])),
#             "saturation_factor": max(1.0, min(1.3, recommendations["saturation_factor"]))
#         }
#     except Exception as e:
#         logging.error(f"LLM query failed: {str(e)}")
#         return {
#             "brightness_factor": 1.2,
#             "contrast_factor": 1.3,
#             "sharpness_factor": 0.2,
#             "saturation_factor": 1.15
#         }

# def enhance_image(image, bbox, recommendations):
#     """Enhance product image quality"""
#     x1, y1, x2, y2 = bbox
    
#     # Initial noise reduction
#     product = cv2.fastNlMeansDenoisingColored(
#         image[y1:y2, x1:x2],
#         None,
#         h=10,
#         hColor=10,
#         templateWindowSize=7,
#         searchWindowSize=21
#     )
    
#     # Convert to LAB for better processing
#     lab = cv2.cvtColor(product, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
    
#     # Enhance lighting with CLAHE
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     l = clahe.apply(l)
    
#     # Adjust contrast and brightness
#     l = cv2.convertScaleAbs(
#         l,
#         alpha=recommendations["contrast_factor"],
#         beta=int((recommendations["brightness_factor"] - 1.0) * 127)
#     )
    
#     # Enhance color
#     a = cv2.convertScaleAbs(a, alpha=recommendations["saturation_factor"])
#     b = cv2.convertScaleAbs(b, alpha=recommendations["saturation_factor"])
    
#     # Merge and convert back
#     enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
#     # Apply bilateral filter for edge preservation
#     enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
#     # Create white background
#     target_size = 2000
#     square = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    
#     # Scale for 85% frame fill
#     scale = min(target_size / enhanced.shape[0], target_size / enhanced.shape[1]) * 0.85
#     new_size = (int(enhanced.shape[1] * scale), int(enhanced.shape[0] * scale))
#     enhanced = cv2.resize(enhanced, new_size, interpolation=cv2.INTER_CUBIC)
    
#     # Center the product
#     y_offset = (target_size - enhanced.shape[0]) // 2
#     x_offset = (target_size - enhanced.shape[1]) // 2
#     square[y_offset:y_offset + enhanced.shape[0], 
#            x_offset:x_offset + enhanced.shape[1]] = enhanced
    
#     return square


import cv2
import numpy as np
import torch
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

# Initialize
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def detect_object(image):
    """Detect product and ensure 85% frame fill."""
    logger.info("Starting object detection.")
    results = model(image[..., ::-1])

    detections = results.xyxy[0].cpu().numpy()
    if len(detections) == 0:
        logger.warning("No objects detected, using full image.")
        height, width, _ = image.shape
        return (0, 0, width, height), image

    largest_object = max(detections, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
    x1, y1, x2, y2 = map(int, largest_object[:4])

    # Ensure 85% frame fill
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    size = int(max(x2 - x1, y2 - y1) / 0.85)

    x1 = max(0, center_x - size // 2)
    y1 = max(0, center_y - size // 2)
    x2 = min(image.shape[1], center_x + size // 2)
    y2 = min(image.shape[0], center_y + size // 2)

    logger.info(f"Object detected at coordinates: ({x1}, {y1}, {x2}, {y2}).")
    return (x1, y1, x2, y2), image[y1:y2, x1:x2]

def extract_features(image):
    """Extract image quality metrics."""
    logger.info("Extracting image features.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    features = {
        "brightness": np.mean(gray),
        "contrast": np.std(gray),
        "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var(),
        "color_balance": {
            "l": np.mean(lab[:, :, 0]),
            "a": np.mean(lab[:, :, 1]),
            "b": np.mean(lab[:, :, 2])
        }
    }
    logger.info(f"Extracted features: {features}.")
    return features

def query_llm(features):
    """Get enhancement parameters from LLM."""
    logger.info("Querying LLM for enhancement parameters.")
    prompt = f"""
    Analyze these product photo metrics and recommend optimal enhancement values:

    Current Image Metrics:
    - Brightness: {features['brightness']:.1f} (optimal range: 180-220)
    - Contrast: {features['contrast']:.1f} (optimal range: 40-60)
    - Sharpness: {features['sharpness']:.1f} (optimal range: 800-1200)
    - Color Balance L: {features['color_balance']['l']:.1f}
    - Color Balance a: {features['color_balance']['a']:.1f}
    - Color Balance b: {features['color_balance']['b']:.1f}

    Provide enhancement parameters as a Python dictionary:
    {{
        "brightness_factor": (float, e.g., 1.1),
        "contrast_factor": (float, e.g., 1.2),
        "sharpness_factor": (float, e.g., 0.2),
        "saturation_factor": (float, e.g., 1.3)
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in product photo retouching. Respond only with valid Python dictionary syntax."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        recommendations = eval(response.choices[0].message.content.strip())
        logger.info(f"LLM recommendations: {recommendations}.")
        return recommendations
    except Exception as e:
        logger.error(f"LLM query failed: {str(e)}.")
        # Default fallback parameters
        return {
            "brightness_factor": 1.0,
            "contrast_factor": 1.1,
            "sharpness_factor": 0.3,
            "saturation_factor": 1.0
        }

def enhance_image(image, bbox, recommendations):
    """Enhance product image quality."""
    logger.info("Applying image enhancements.")
    x1, y1, x2, y2 = bbox
    product = image[y1:y2, x1:x2]

    # Convert to LAB color space for better adjustments
    lab = cv2.cvtColor(product, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Enhance brightness and contrast
    l = cv2.convertScaleAbs(l, alpha=recommendations["contrast_factor"], beta=int((recommendations["brightness_factor"] - 1.0) * 127))

    # Adjust saturation
    a = cv2.convertScaleAbs(a, alpha=recommendations["saturation_factor"])
    b = cv2.convertScaleAbs(b, alpha=recommendations["saturation_factor"])

    # Merge channels and convert back to BGR
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Apply sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * recommendations["sharpness_factor"]
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    logger.info("Image enhancement complete.")
    return enhanced

def process_image(image_path):
    """Main function to process and enhance an image."""
    logger.info(f"Processing image: {image_path}.")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}.")
        return None

    # Detect object and get bounding box
    bbox, _ = detect_object(image)

    # Extract features and query LLM for recommendations
    features = extract_features(image)
    recommendations = query_llm(features)

    # Apply enhancements
    enhanced_image = enhance_image(image, bbox, recommendations)

    # Save enhanced image
    output_path = f"enhanced_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, enhanced_image)
    logger.info(f"Enhanced image saved at {output_path}.")
    return output_path


