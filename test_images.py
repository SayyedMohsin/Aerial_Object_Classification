import numpy as np
from PIL import Image
import os

def test_bird_image():
    """Test with actual bird image from dataset"""
    bird_path = r"C:\JN\Aerial_Object_Classification\data\classification_dataset\train\bird"
    
    if os.path.exists(bird_path):
        bird_images = [f for f in os.listdir(bird_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if bird_images:
            # Test first 5 bird images
            for i in range(min(5, len(bird_images))):
                img_path = os.path.join(bird_path, bird_images[i])
                try:
                    image = Image.open(img_path)
                    img_array = np.array(image)
                    
                    print(f"\n🧪 Testing BIRD image {i+1}: {bird_images[i]}")
                    print(f"   Size: {image.size}")
                    print(f"   Array shape: {img_array.shape}")
                    
                    # Calculate features
                    if len(img_array.shape) == 3:
                        print(f"   Avg brightness: {np.mean(img_array):.1f}")
                        print(f"   Color variance: {np.var(img_array):.0f}")
                        
                        # Check if it's dark or bright
                        if np.mean(img_array) < 100:
                            print("   ⚠️ Dark image - might be classified as DRONE")
                        else:
                            print("   ✅ Bright image - should be BIRD")
                    
                except Exception as e:
                    print(f"   Error: {e}")

def test_drone_image():
    """Test with actual drone image from dataset"""
    drone_path = r"C:\JN\Aerial_Object_Classification\data\classification_dataset\train\drone"
    
    if os.path.exists(drone_path):
        drone_images = [f for f in os.listdir(drone_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if drone_images:
            # Test first 5 drone images
            for i in range(min(5, len(drone_images))):
                img_path = os.path.join(drone_path, drone_images[i])
                try:
                    image = Image.open(img_path)
                    img_array = np.array(image)
                    
                    print(f"\n🧪 Testing DRONE image {i+1}: {drone_images[i]}")
                    print(f"   Size: {image.size}")
                    print(f"   Array shape: {img_array.shape}")
                    
                    # Calculate features
                    if len(img_array.shape) == 3:
                        print(f"   Avg brightness: {np.mean(img_array):.1f}")
                        print(f"   Color variance: {np.var(img_array):.0f}")
                        
                        # Check if it's dark or bright
                        if np.mean(img_array) < 100:
                            print("   ⚠️ Dark image - will be classified as DRONE")
                        else:
                            print("   ✅ Bright image - might be classified as BIRD")
                    
                except Exception as e:
                    print(f"   Error: {e}")

def main():
    print("🔍 DEBUGGING IMAGE PROPERTIES...")
    print("="*50)
    
    print("\n📸 TESTING BIRD IMAGES:")
    test_bird_image()
    
    print("\n" + "="*50)
    
    print("\n📸 TESTING DRONE IMAGES:")
    test_drone_image()
    
    print("\n" + "="*50)
    print("\n🎯 ANALYSIS:")
    print("If BIRD images are dark (avg brightness < 100), they'll be classified as DRONE")
    print("If DRONE images are bright (avg brightness > 150), they'll be classified as BIRD")
    print("\n💡 SOLUTION: Adjust brightness thresholds in the logic")

if __name__ == "__main__":
    main()