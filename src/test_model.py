import os
import tensorflow as tf
import numpy as np
from PIL import Image

print("🔍 TESTING CURRENT MODEL...")

# Check model file
model_path = "models/final_model.h5"
if os.path.exists(model_path):
    print(f"✅ Model found: {model_path} ({os.path.getsize(model_path)/(1024*1024):.1f} MB)")
else:
    print(f"❌ Model NOT found: {model_path}")
    exit()

# Load model
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully")

# Test with ONE known image
test_images = [
    r"C:\JN\Aerial_Object_Classification\data\classification_dataset\test\drone\foto01915_png.rf.7d7cd852392707f519d13e9cf051de3f.jpg",
    r"C:\JN\Aerial_Object_Classification\data\classification_dataset\test\bird\00083b384685315d_jpg.rf.abfd1b2cc8c681777bae66d5327bb9ea.jpg"
]

for img_path in test_images:
    if os.path.exists(img_path):
        print(f"\n🧪 Testing: {os.path.basename(img_path)}")
        print(f"   Expected: {'DRONE' if 'drone' in img_path else 'BIRD'}")
        
        try:
            # Load and preprocess image
            img = Image.open(img_path)
            img = img.resize((150, 150))
            img_array = np.array(img) / 255.0
            
            if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_array, verbose=0)
            raw_score = float(prediction[0][0])
            
            print(f"   Raw score: {raw_score:.4f}")
            
            # Current logic
            if raw_score > 0.5:
                predicted = "DRONE"
                confidence = raw_score
            else:
                predicted = "BIRD" 
                confidence = 1 - raw_score
                
            print(f"   Predicted: {predicted}")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   Status: {'✅' if ('drone' in img_path and predicted=='DRONE') or ('bird' in img_path and predicted=='BIRD') else '❌'}")
            
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print(f"❌ Image not found: {img_path}")

print("\n🎯 ANALYSIS:")
print("If DRONE images show score < 0.5, we need to FLIP the logic")
input("\nPress Enter to exit...")