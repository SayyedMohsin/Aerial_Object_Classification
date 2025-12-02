import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

print("🔥 Creating ULTRA-COMPATIBLE Model...")

def create_simple_compatible_model():
    """Create model that works on ALL TensorFlow versions"""
    
    # SUPER SIMPLE model - no Input layer issues
    model = keras.Sequential()
    
    # Add Conv2D with input_shape parameter (NOT Input layer)
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(keras.layers.MaxPooling2D(2, 2))
    
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(2, 2))
    
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
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
    
    # Create model
    model = create_simple_compatible_model()
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Save in formats
    os.makedirs('../models', exist_ok=True)
    
    # Save as SavedModel format (most compatible)
    model.save('../models/compatible_model', save_format='tf')
    
    # Also save as .h5
    model.save('../models/compatible_model.h5')
    
    print("\n✅ Models saved:")
    print("   - models/compatible_model/ (SavedModel format)")
    print("   - models/compatible_model.h5 (H5 format)")
    
    # Test prediction
    print("\n🧪 Testing model with dummy input...")
    dummy_input = np.random.random((1, 150, 150, 3)).astype(np.float32)
    prediction = model.predict(dummy_input, verbose=0)
    print(f"   Dummy prediction: {prediction[0][0]:.4f}")
    print("   ✅ Model is working!")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")