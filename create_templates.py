import os
import numpy as np
from PIL import Image
import pickle

print("📁 Creating Template Database for Perfect Matching...")

def create_template_database():
    """Create templates from training data"""
    data_path = r"C:\JN\Aerial_Object_Classification\data\classification_dataset\train"
    
    templates = {
        'bird': [],
        'drone': []
    }
    
    # Create bird templates
    bird_path = os.path.join(data_path, 'bird')
    bird_images = [f for f in os.listdir(bird_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:50]
    
    for img_name in bird_images:
        try:
            img_path = os.path.join(bird_path, img_name)
            img = Image.open(img_path).resize((100, 100))
            img_array = np.array(img) / 255.0
            templates['bird'].append(img_array)
            print(f"✅ Added bird template: {img_name}")
        except Exception as e:
            print(f"❌ Error with {img_name}: {e}")
    
    # Create drone templates  
    drone_path = os.path.join(data_path, 'drone')
    drone_images = [f for f in os.listdir(drone_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:50]
    
    for img_name in drone_images:
        try:
            img_path = os.path.join(drone_path, img_name)
            img = Image.open(img_path).resize((100, 100))
            img_array = np.array(img) / 255.0
            templates['drone'].append(img_array)
            print(f"✅ Added drone template: {img_name}")
        except Exception as e:
            print(f"❌ Error with {img_name}: {e}")
    
    # Save templates
    with open('../templates.pkl', 'wb') as f:
        pickle.dump(templates, f)
    
    print(f"\n🎉 Template Database Created:")
    print(f"   Bird templates: {len(templates['bird'])}")
    print(f"   Drone templates: {len(templates['drone'])}")
    print(f"   Saved as: templates.pkl")

if __name__ == "__main__":
    create_template_database()