import os
import shutil

print("🚀 Preparing Hugging Face Deployment Package...")

# Create deployment folder
deploy_folder = "huggingface_deploy"
if os.path.exists(deploy_folder):
    shutil.rmtree(deploy_folder)

os.makedirs(deploy_folder)
os.makedirs(os.path.join(deploy_folder, "models"))

# Copy files
files_to_copy = [
    ("app_huggingface.py", "app.py"),
    ("requirements_hf.txt", "requirements.txt"),
    ("models/hf_model.keras", "models/hf_model.keras"),
    ("models/hf_model.h5", "models/hf_model.h5")
]

for src, dst in files_to_copy:
    if os.path.exists(src):
        shutil.copy(src, os.path.join(deploy_folder, dst))
        print(f"✅ Copied: {src} → {dst}")

# Create README
readme_content = """---
title: Aerial Object Classification
emoji: 🛸
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# 🛸 Aerial Intelligence Platform

Real-time Bird vs Drone classification using Deep Learning AI.

## Features
- 🐦 Bird Detection
- 🚁 Drone Detection
- 📊 High Accuracy
- 🎨 Professional UI
- ⚡ Real-time Analysis

## How to Use
1. Upload an aerial image
2. AI analyzes automatically
3. View results with confidence score

## Technology
- TensorFlow 2.15
- Streamlit
- Computer Vision AI
"""

with open(os.path.join(deploy_folder, "README.md"), "w") as f:
    f.write(readme_content)
print("✅ Created: README.md")

print(f"\n🎉 Deployment package ready in: {deploy_folder}/")
print("\n📁 Upload these files to Hugging Face:")
for root, dirs, files in os.walk(deploy_folder):
    for file in files:
        print(f"  - {os.path.join(root, file)}")