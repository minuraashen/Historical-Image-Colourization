
# **🌈 Intelligent Historical Image Restoration** (Just Started)
## Deep Learning for Colorization, Denoising & Super-Resolution with Explainability
***Currently initiating this project to explore and apply deep learning and image processing techniques for colorization, denoising, and super-resolution, while gaining hands-on experience with explainable AI in computer vision. Proposed features and design flow are described below.***
<!-- Add a sample result image here -->

## **📌 Overview**
### **This project restores old or grayscale images by applying:**
✔ Automatic Colorization using CNN + GAN  
✔ Scratch Removal & Denoising  
✔ Super-Resolution Enhancement with ESRGAN  
✔ Explainability (Attention Heatmaps using Grad-CAM)  
✔ Web-based Deployment for user interaction  

## **✨ Features**
✅ Automatic Colorization – Converts grayscale to realistic color  
✅ Denoising & Scratch Removal – Removes noise and artifacts  
✅ Super-Resolution – Enhances image quality for sharp details  
✅ Explainable AI – Visualize attention maps for colorization decisions  
✅ Web App – Upload and restore photos instantly  
✅ Edge Optimized – Model converted to ONNX and quantized for deployment on low-power devices  

## **🖼 Sample Results**
Input	Colorized	Restored + Super-Resolved

## **🏗 Project Architecture**
Input Image (Grayscale)  
    ↓  
[Colorization Model] → Adds realistic color  
    ↓  
[Denoising Autoencoder] → Removes noise and scratches  
    ↓  
[Super-Resolution (ESRGAN)] → Improves quality  
    ↓  
[Explainability Layer] → Attention heatmaps  
    ↓  
Output Image (Restored & Colorized)  


## **🔍 Dataset**
***Primary Sources:***
- ImageNet  
- COCO Dataset  
- Kaggle Historical Images  




