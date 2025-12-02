import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

print("🔍 DEBUGGING MODEL PREDICTIONS...")

# Load the trained model
model_path = "../models/best_model.h5"
if not os.path.exists(model_path):
    print("❌ Model file not found! Please run training first.")
    exit()

model = keras.models.load_model(model_path)
print("✅ Model loaded successfully")

# Test with actual dataset images
data_path = r"C:\JN\Aerial_Object_Classification\data\classification_dataset"

def test_images(folder_path, expected_class):
    if not os.path.exists(folder_path):
        print(f"❌ Path not found: {folder_path}")
        return
    
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
    print(f"\n🧪 Testing {len(images)} {expected_class.upper()} images:")
    
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path)
            img = img.resize((150, 150))
            img_array = np.array(img) / 255.0
            
            if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array, verbose=0)
            confidence = float(prediction[0][0])
            
            # Current understanding: >0.5 = Drone, <0.5 = Bird
            predicted_class = "DRONE" if confidence > 0.5 else "BIRD"
            
            print(f"   {img_name}:")
            print(f"      Raw score: {confidence:.4f}")
            print(f"      Predicted: {predicted_class}")
            print(f"      Expected:  {expected_class.upper()}")
            print(f"      Status:    {'✅ CORRECT' if predicted_class == expected_class.upper() else '❌ WRONG'}")
            
        except Exception as e:
            print(f"   Error with {img_name}: {e}")

# Test drone images
drone_test_path = os.path.join(data_path, "test", "drone")
test_images(drone_test_path, "drone")

# Test bird images  
bird_test_path = os.path.join(data_path, "test", "bird")
test_images(bird_test_path, "bird")

print("\n🎯 ANALYSIS:")
print("If DRONE images show score < 0.5 and BIRD images show score > 0.5,")
print("then we have CLASS LABEL INVERSION in the model!")