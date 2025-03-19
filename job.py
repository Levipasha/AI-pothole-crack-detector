import cv2
import json
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, InterpolationMode
from PIL import Image  # Import PIL for image conversion
from inference_sdk import InferenceHTTPClient
import matplotlib.pyplot as plt

# Set the new image path
image_path = r"C:"#enter your image here with path 

# Set up the client
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="wrtBN2vi1PaoJO9I9goq"
)

# Perform inference
result = CLIENT.infer(image_path, model_id="road-potholes-and-cracks-jmxtp/3")

# Check for inference result
if not result or 'predictions' not in result:
    print("No predictions were returned. Please check your model or input image.")
    exit()

# Print the result to inspect its structure
print(json.dumps(result, indent=4))  # Pretty print the result

# Load the original image
original_image = cv2.imread(image_path)  # Ensure this matches the inference path
if original_image is None:
    print("Error loading image. Please check the file path.")
    exit()

# Convert the original image from BGR (OpenCV format) to RGB (PIL format)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_pil = Image.fromarray(original_image_rgb)  # Convert to PIL Image

# Get confidence threshold from user input
try:
    confidence_threshold = float(input("Enter confidence threshold (0.0 to 1.0): "))
except ValueError:
    print("Invalid input! Setting confidence threshold to 0.5.")
    confidence_threshold = 0.5

# Ensure the threshold is within the valid range
if confidence_threshold < 0.0 or confidence_threshold > 1.0:
    print("Confidence threshold must be between 0.0 and 1.0. Setting it to 0.5.")
    confidence_threshold = 0.5

# Load the MiDaS model for depth estimation
depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
depth_model.eval()

# Initialize transforms for depth model
transform = Compose([
    Resize((384, 384), interpolation=InterpolationMode.BILINEAR),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize counters and lists for detected classes, depths, and areas
detected_count = 0
detected_classes = []
depth_estimates = []
area_estimates_cm2 = []  # List to hold areas of detected cracks and potholes in cm²

# Define scale in pixels per cm (for example, 10 pixels/cm)
scale = 10  

# Process the result to draw predictions
for prediction in result.get('predictions', []):
    confidence = prediction.get('confidence', 0)

    # Skip predictions below the confidence threshold
    if confidence < confidence_threshold:
        continue  

    # Check for different key structures for bounding box
    if 'x' in prediction and 'y' in prediction and 'width' in prediction and 'height' in prediction:
        x_center = int(prediction['x'])
        y_center = int(prediction['y'])
        width = int(prediction['width'])
        height = int(prediction['height'])

        # Calculate bounding box coordinates
        x1 = max(0, x_center - (width // 2))  # Ensure x1 is not negative
        y1 = max(0, y_center - (height // 2))  # Ensure y1 is not negative
        x2 = min(original_image.shape[1], x_center + (width // 2))  # Ensure x2 is within image width
        y2 = min(original_image.shape[0], y_center + (height // 2))  # Ensure y2 is within image height

    elif 'x1' in prediction and 'y1' in prediction and 'x2' in prediction and 'y2' in prediction:
        x1 = int(prediction['x1'])
        y1 = int(prediction['y1'])
        x2 = int(prediction['x2'])
        y2 = int(prediction['y2'])
    else:
        continue  # Skip this prediction if it doesn't match expected structure

    label = prediction.get('class', 'Unknown')  # Adjust according to your model's output
    detected_classes.append(label)
    detected_count += 1

    # Calculate the area of the detected region
    area_pixels = width * height  # Area in pixels

    # Convert area from pixels to cm²
    area_cm2 = area_pixels / (scale ** 2)  # Area in cm²
    area_estimates_cm2.append(area_cm2)

    # Perform depth estimation on the cropped region
    cropped_image = original_image[y1:y2, x1:x2]
    if cropped_image.size > 0:
        # Convert cropped image to PIL format
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image_pil = Image.fromarray(cropped_image_rgb)

        # Prepare the cropped image for depth model
        input_tensor = transform(cropped_image_pil).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            depth_map = depth_model(input_tensor)

        # Process depth_map to get average depth in cm
        average_depth = depth_map.mean().item()  # Average depth in meters
        depth_in_cm = average_depth * 100  # Convert to cm
        depth_estimates.append(depth_in_cm)

    # Draw bounding box
    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(original_image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Save the resulting image
output_path = "C:/Users/vamsh/Downloads/fwnewwork/fusion-wear1-master/result_image.jpg"
cv2.imwrite(output_path, original_image)

# Print detected count and classes
print(f"Total detected: {detected_count}")
for cls in set(detected_classes):  # Print unique classes
    count = detected_classes.count(cls)
    print(f"{cls}: {count}")

# Print estimated depths in mm
print("Estimated Depths (in mm):")
for depth in depth_estimates:
    print(f"{depth:.2f} mm")

# Print areas of detected cracks and potholes in mm²
print("Areas of detected cracks and potholes (in mm):")
for area in area_estimates_cm2:
    print(f"{area:.2f} mm")

# Optionally, display the result using Matplotlib
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()
