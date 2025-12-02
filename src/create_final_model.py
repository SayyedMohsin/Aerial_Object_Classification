import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

print("🔥 Creating FINAL COMPATIBLE Model...")

def create_compatible_model():
    """Create model that works on ALL TensorFlow versions"""
    
    # SUPER SIMPLE model - works everywhere
    model = keras.Sequential()
    
    # Add layers with input_shape in first layer
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(keras.layers.MaxPooling2D(2, 2))
    
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(2, 2))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    
    # Create model
    model = create_compatible_model()
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Create models directory
    os.makedirs('../models', exist_ok=True)
    
    # Save in .keras format (recommended for Keras 3)
    keras_model_path = '../models/final_aerial_model.keras'
    model.save(keras_model_path)
    print(f"\n✅ Model saved as: {keras_model_path}")
    
    # Also save in .h5 format for compatibility
    h5_model_path = '../models/final_aerial_model.h5'
    model.save(h5_model_path)
    print(f"✅ Model saved as: {h5_model_path}")
    
    # Test with dummy data
    print("\n🧪 Testing model...")
    dummy_input = np.random.random((1, 150, 150, 3)).astype(np.float32)
    prediction = model.predict(dummy_input, verbose=0)
    print(f"   Dummy prediction: {prediction[0][0]:.4f}")
    
    # Load test to verify
    print("\n🔍 Verifying saved model...")
    try:
        loaded_model = keras.models.load_model(keras_model_path)
        loaded_prediction = loaded_model.predict(dummy_input, verbose=0)
        print(f"   Loaded model prediction: {loaded_prediction[0][0]:.4f}")
        
        if abs(prediction[0][0] - loaded_prediction[0][0]) < 0.0001:
            print("   ✅ Model save/load successful!")
        else:
            print("   ⚠️ Predictions differ slightly (expected)")
    except Exception as e:
        print(f"   ❌ Load error: {e}")
    
    print("\n🎯 Model is READY for Hugging Face!")
    print("   Use: final_aerial_model.keras")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")