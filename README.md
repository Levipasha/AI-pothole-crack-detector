# AI-pothole-crack-detector
# 🛣️ Pothole and Crack Detector using YOLOv8 & MiDaS

This project uses **YOLOv8** and **MiDaS depth estimation** to detect **potholes and cracks** on roads. It leverages **Roboflow API** for object detection and **PyTorch** for depth estimation, helping identify road damage and estimating its severity.

---

## 📌 Features

✅ Detects potholes and cracks in images.  
✅ Computes **confidence scores** for detections.  
✅ Estimates **depth** using the MiDaS model.  
✅ Calculates **area** of detected potholes and cracks.  
✅ Saves the processed image with bounding boxes.  
✅ Provides real-time visualization of results.  

---

## ⚡ Technologies Used

- **Python** (Main language)
- **OpenCV** (Image processing)
- **Torch & torchvision** (Machine learning & deep learning)
- **Roboflow API** (Object detection)
- **Matplotlib** (Image visualization)
- **PIL (Pillow)** (Image conversion)

---

## 📂 Project Structure

pip install opencv-python torch torchvision matplotlib inference-sdk pillow
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="YOUR_API_KEY"
)
python job.py
