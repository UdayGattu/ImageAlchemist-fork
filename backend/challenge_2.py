

import cv2
import numpy as np
import os
import logging
from rembg import remove
from transformers import DetrImageProcessor, DetrForObjectDetection
from sklearn.cluster import KMeans
from openai import OpenAI
from dotenv import load_dotenv
from ultralytics import YOLO
import json
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# DETR and YOLO models
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
yolo_model = YOLO("yolov8s.pt")

# Logging setup
LOG_FILE = "image_processing.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

FINAL_IMAGE_SIZE = 2000
PRODUCT_OCCUPANCY = 0.85


#


def detect_with_detr(image):
    """Detect the product in the image using DETR."""
    logging.info("Starting object detection with DETR.")
    inputs = detr_processor(images=image, return_tensors="pt")
    outputs = detr_model(**inputs)

    # Process detections
    logits = outputs.logits.softmax(-1)[0, :, :-1]  # Exclude background
    boxes = outputs.pred_boxes[0]  # Predicted bounding boxes
    max_probs, labels = logits.max(-1).values, logits.argmax(-1)
    confidence_threshold = 0.2
    valid_indices = max_probs > confidence_threshold

    if not valid_indices.any():
        logging.warning("No valid objects detected with DETR.")
        return None, None

    # Select the largest bounding box
    valid_boxes = boxes[valid_indices].detach().cpu().numpy()
    valid_labels = labels[valid_indices].detach().cpu().numpy()
    largest_object_idx = np.argmax(
        [(box[2] - box[0]) * (box[3] - box[1]) for box in valid_boxes]
    )
    x1, y1, x2, y2 = map(int, valid_boxes[largest_object_idx])
    object_type = detr_model.config.id2label[valid_labels[largest_object_idx]]

    logging.info(f"DETR detected bounding box: ({x1}, {y1}, {x2}, {y2})")
    logging.info(f"DETR detected object type: {object_type}")
    return (x1, y1, x2, y2), object_type




def detect_with_yolo(image):
    """Detect the product in the image using YOLO."""
    logging.info("Starting object detection with YOLO.")
    results = yolo_model.predict(source=image, save=False, conf=0.2)

    detections = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
    labels = results[0].names if results else []

    if len(detections) == 0:
        logging.warning("No valid objects detected with YOLO.")
        return None, None

    # Select the largest bounding box
    largest_object_idx = np.argmax(
        [(box[2] - box[0]) * (box[3] - box[1]) for box in detections]
    )

    # Validate the detection structure
    try:
        x1, y1, x2, y2 = map(int, detections[largest_object_idx][:4])
        label_index = int(detections[largest_object_idx][5]) if len(detections[largest_object_idx]) > 5 else None
        object_type = labels[label_index] if label_index is not None and label_index < len(labels) else "unknown"
    except Exception as e:
        logging.error(f"Error processing YOLO detections: {e}")
        return None, None

    logging.info(f"YOLO detected bounding box: ({x1}, {y1}, {x2}, {y2})")
    logging.info(f"YOLO detected object type: {object_type}")
    return (x1, y1, x2, y2), object_type


def fallback_detection(image):
    """Fallback detection method using contour detection."""
    logging.info("Fallback detection using contours.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logging.warning("No contours found in the image.")
        return None, "unknown object"

    # Select the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    logging.info(f"Contour detected bounding box: ({x}, {y}, {x + w}, {y + h})")
    return (x, y, x + w, y + h), "unknown object"


def detect_product(image):
    """Detect the product using multiple methods."""
    bbox, object_type = detect_with_detr(image)
    if not bbox or bbox == (0, 0, 0, 0):
        logging.info("Fallback to YOLO for object detection.")
        bbox, object_type = detect_with_yolo(image)
    if not bbox or bbox == (0, 0, 0, 0):
        logging.info("Fallback to contour-based detection.")
        bbox, object_type = fallback_detection(image)

    if not bbox or bbox == (0, 0, 0, 0):
        raise ValueError("No valid bounding box detected with any method.")

    x1, y1, x2, y2 = bbox
    cropped = image[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
    logging.info(f"Final detected bounding box: ({x1}, {y1}, {x2}, {y2})")
    logging.info(f"Detected object type: {object_type}")
    return bbox, cropped, object_type


def remove_background(image):
    """Remove the background of the product."""
    try:
        _, buffer = cv2.imencode(".png", image)
        transparent_product = remove(buffer.tobytes())
        decoded_image = cv2.imdecode(np.frombuffer(transparent_product, np.uint8), cv2.IMREAD_UNCHANGED)
        return decoded_image
    except Exception as e:
        logging.error(f"Error removing background: {e}")
        raise


def extract_dominant_color(image, k=3):
    """Extract the dominant color from the image using KMeans clustering."""
    pixels = image[:, :, :3].reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    return [int(c) for c in dominant_color]


def query_llm(features, object_type, background_type):
    """Query LLM for background recommendations based on product features."""
    prompt = f"""
    The product is a {object_type} with a dominant color {features['color']} and is intended for a background type "{background_type}".
    As an eCommerce photo optimization expert, your goal is to recommend visually appealing and diverse background color palettes
    that make the product stand out. Please provide recommendations with the following considerations:

    1. **Color**:
       Suggest an RGB value that contrasts the dominant color of the product. The color background should enhance the visibility
       and appeal of the product while maintaining a clean and professional look. For example, use complementary or analogous colors.

    2. **Gradient Background**:
       Provide two RGB values for a gradient background. The gradient should create a smooth and visually pleasing transition, focusing
       on tones that are either complementary to the product's dominant color or convey a specific mood (e.g., vibrant, minimalistic, luxurious).

    3. **Application Context**:
       These recommendations should account for common eCommerce use cases:
       - For fashion products, use warm or neutral tones that emphasize texture and details.
       - For electronics, opt for cool or futuristic tones like blues and silvers.
       - For furniture, use earth tones or soft gradients to convey comfort and style.
       - For {object_type}, ensure the background highlights its unique characteristics and makes it stand out in a professional manner.

    4. **Consider Trends**:
       Incorporate current eCommerce and design trends, such as pastel gradients, monochromatic themes, or bold contrasting colors.

    5. **Output Format**:
       {{
           "background_type": "{background_type}",
           "solid_color": [<R>, <G>, <B>],
           "gradient_start": [<R>, <G>, <B>],
           "gradient_end": [<R>, <G>, <B>]
       }}

    Ensure the response adheres to JSON format. Example:
    {{
        "background_type": "Solid Color",
        "solid_color": [255, 255, 255],
        "gradient_start": [240, 240, 240],
        "gradient_end": [200, 200, 200]
    }}

    Use your knowledge of visual design, color psychology, and eCommerce optimization to generate precise and impactful suggestions.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in eCommerce product photo optimization."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        match = re.search(r"\{.*?\}", response.choices[0].message.content, re.DOTALL)
        if not match:
            raise ValueError("LLM response does not contain valid JSON.")
        recommendations = json.loads(match.group(0))
        logging.info(f"Recommendations: {recommendations}")
        return recommendations
    except Exception as e:
        logging.error(f"Error querying LLM: {e}")
        raise

def resize_and_center_product(product, canvas_size):
    """Resize and center the product image to fit within the canvas."""
    product_h, product_w = product.shape[:2]
    target_size = int(PRODUCT_OCCUPANCY * canvas_size)
    scale = min(target_size / product_h, target_size / product_w)
    resized_product = cv2.resize(product, (int(product_w * scale), int(product_h * scale)), interpolation=cv2.INTER_LINEAR)

    # Create a blank canvas with alpha channel
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    y_offset = (canvas_size - resized_product.shape[0]) // 2
    x_offset = (canvas_size - resized_product.shape[1]) // 2

    # Overlay the resized product on the canvas
    canvas[y_offset:y_offset + resized_product.shape[0], x_offset:x_offset + resized_product.shape[1]] = resized_product
    return canvas


def add_background(product, canvas_size, params, background_type):
    """Generate a background based on the type and parameters."""
    if background_type == "Solid Color":
        color = params.get("solid_color", [255, 255, 255])  # Default to white
        logging.info(f"Using solid color: {color}")
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)
    elif background_type == "Gradient":
        gradient_start = params.get("gradient_start", [255, 255, 255])
        gradient_end = params.get("gradient_end", [0, 0, 0])
        logging.info(f"Using gradient: start={gradient_start}, end={gradient_end}")

        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        for y in range(canvas_size):
            ratio = y / canvas_size
            color = [
                int(gradient_start[i] * (1 - ratio) + gradient_end[i] * ratio) for i in range(3)
            ]
            canvas[y, :] = np.array(color, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported background type: {background_type}")

    # Resize and center the product
    product = resize_and_center_product(product, canvas_size)

    # Blend the product with the background
    alpha_channel = product[:, :, 3] / 255.0
    for c in range(3):
        canvas[:, :, c] = (product[:, :, c] * alpha_channel + canvas[:, :, c] * (1 - alpha_channel)).astype(np.uint8)

    return canvas

def add_studio_setting_background(product, canvas_size, params):
    """Generate a studio setting background with LLM recommendations and soft lighting."""
    logging.info("Generating studio setting background.")

    # Use LLM-recommended gradient start and end colors
    gradient_start = params.get("gradient_start", [240, 240, 240])  # Default light gray start
    gradient_end = params.get("gradient_end", [200, 200, 200])  # Default darker gray end
    logging.info(f"Using LLM-recommended gradient: start={gradient_start}, end={gradient_end}")

    # Create a gradient background
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    for y in range(canvas_size):
        ratio = y / canvas_size
        color = [
            int(gradient_start[i] * (1 - ratio) + gradient_end[i] * ratio) for i in range(3)
        ]
        canvas[y, :] = np.array(color, dtype=np.uint8)

    # Blend the product with the generated background
    return blend_product_with_background(product, canvas)



def add_simple_lifestyle_context(product, canvas_size, params, texture_path="lifestyle_texture.jpg"):
    """Generate a simple lifestyle context background using LLM recommendations."""
    logging.info("Generating simple lifestyle context background.")

    # Check if texture file exists
    if os.path.exists(texture_path):
        texture = cv2.imread(texture_path, cv2.IMREAD_COLOR)
        texture = cv2.resize(texture, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)
        logging.info(f"Using lifestyle texture: {texture_path}")
    else:
        logging.warning(f"Texture file {texture_path} not found. Using LLM-recommended gradient.")
        gradient_start = params.get("gradient_start", [255, 255, 255])  # Default white start
        gradient_end = params.get("gradient_end", [240, 240, 240])  # Default light gray end

        # Create a fallback gradient
        texture = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        for y in range(canvas_size):
            ratio = y / canvas_size
            color = [
                int(gradient_start[i] * (1 - ratio) + gradient_end[i] * ratio) for i in range(3)
            ]
            texture[y, :] = np.array(color, dtype=np.uint8)

    # Blend the product with the generated background or texture
    return blend_product_with_background(product, texture)



def blend_product_with_background(product, background):
    """Blend the product with the background, adding shadows and reflections."""
    logging.info("Blending product with background.")
    canvas_size = background.shape[0]

    # Ensure product dimensions match canvas dimensions
    if product.shape[0] != canvas_size or product.shape[1] != canvas_size:
        logging.info(f"Resizing product from {product.shape[:2]} to {canvas_size}.")
        product = cv2.resize(product, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)

    # Create a shadow effect based on the product's alpha channel
    shadow = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    shadow[:, :, 3] = cv2.GaussianBlur((product[:, :, 3] > 0).astype(np.uint8) * 255, (51, 51), 0)

    # Apply shadow to the background
    for c in range(3):
        background[:, :, c] = (shadow[:, :, 3] / 255.0 * 50 + background[:, :, c] * (1 - shadow[:, :, 3] / 255.0)).astype(np.uint8)

    # Blend product with the background using alpha channel
    alpha_channel = product[:, :, 3] / 255.0
    for c in range(3):
        background[:, :, c] = (product[:, :, c] * alpha_channel + background[:, :, c] * (1 - alpha_channel)).astype(np.uint8)

    return background



def generate_background_variation(image, background_type, output_dir):
    """Main function to process and create a background variation."""
    logging.info("Starting the background variation generation process.")

    # Step 1: Detect the product in the image and its type
    try:
        bbox, cropped, object_type = detect_product(image)
        logging.info(f"Detected object type: {object_type}")
        
        # Validate bounding box
        if bbox == (0, 0, 0, 0) or cropped is None:
            logging.warning("Invalid bounding box detected. Attempting fallback detection methods.")
            raise ValueError("No valid object detected in the image.")
    except Exception as e:
        logging.error(f"Error during product detection: {e}")
        raise ValueError("Failed to detect product in the image. Ensure the image contains a detectable object.") from e

    # Step 2: Remove the background to isolate the product
    try:
        transparent_product = remove_background(cropped)
    except Exception as e:
        logging.error(f"Error during background removal: {e}")
        raise ValueError("Failed to remove background. Ensure the input image is valid.") from e

    # Step 3: Extract the dominant color from the product
    try:
        dominant_color = extract_dominant_color(transparent_product[:, :, :3])  # Use only RGB channels
        logging.info(f"Dominant color extracted: {dominant_color}")
    except Exception as e:
        logging.error(f"Error during dominant color extraction: {e}")
        raise ValueError("Failed to extract dominant color.") from e

    # Step 4: Query the LLM for background recommendations
    try:
        recommendations = query_llm({"color": dominant_color}, object_type, background_type)
        logging.info(f"LLM recommendations: {recommendations}")
    except Exception as e:
        logging.error(f"Error querying LLM: {e}")
        raise ValueError("Failed to query LLM for background recommendations.") from e

    # Step 5: Generate the background based on the selected type
    try:
        if background_type == "Solid Color":
            final_image = add_background(transparent_product, FINAL_IMAGE_SIZE, recommendations, "Solid Color")
        elif background_type == "Gradient":
            final_image = add_background(transparent_product, FINAL_IMAGE_SIZE, recommendations, "Gradient")
        elif background_type == "Studio Setting":
            final_image = add_studio_setting_background(transparent_product, FINAL_IMAGE_SIZE, recommendations)
        elif background_type == "Simple Lifestyle Context":
            final_image = add_simple_lifestyle_context(transparent_product, FINAL_IMAGE_SIZE, recommendations)
        else:
            raise ValueError(f"Unsupported background type: {background_type}")
    except Exception as e:
        logging.error(f"Error generating background: {e}")
        raise ValueError("Failed to generate background.") from e

    # Step 6: Save the final image
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{background_type.replace(' ', '_').lower()}.jpg")
        cv2.imwrite(output_path, final_image)
        logging.info(f"Generated background saved at {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error saving final image: {e}")
        raise ValueError("Failed to save final image.") from e
