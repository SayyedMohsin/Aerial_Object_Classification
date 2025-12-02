import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("💾 SAVING TRAINED MODEL PROPERLY...")

def create_and_load_model():
    """Create model architecture and load trained weights"""
    
    # Create the same architecture as training
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # First check if we have a trained model
    weights_path = '../models/proper_model_weights.weights.h5'
    
    if os.path.exists(weights_path):
        print(f"✅ Found trained weights: {weights_path}")
        
        # Create model
        model = create_and_load_model()
        
        # Load weights
        model.load_weights(weights_path)
        print("✅ Weights loaded successfully!")
        
        # Save in ALL formats for compatibility
        os.makedirs('../models', exist_ok=True)
        
        # 1. Save as .h5 (full model)
        h5_path = '../models/final_model.h5'
        model.save(h5_path)
        print(f"✅ Full model saved: {h5_path}")
        
        # 2. Save as .keras (new format)
        keras_path = '../models/final_model.keras'
        model.save(keras_path)
        print(f"✅ Keras format saved: {keras_path}")
        
        # 3. Save weights in new format
        new_weights_path = '../models/model_weights.h5'
        model.save_weights(new_weights_path)
        print(f"✅ Weights saved: {new_weights_path}")
        
        # 4. Save model architecture as JSON
        model_json = model.to_json()
        with open('../models/model_architecture.json', 'w') as f:
            f.write(model_json)
        print(f"✅ Architecture saved: ../models/model_architecture.json")
        
        # Test the saved model
        print("\n🧪 Testing saved model...")
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        
        # Test h5 model
        h5_model = keras.models.load_model(h5_path)
        h5_pred = h5_model.predict(dummy_input, verbose=0)
        print(f"   H5 model prediction: {h5_pred[0][0]:.4f}")
        
        # Test keras model
        keras_model = keras.models.load_model(keras_path)
        keras_pred = keras_model.predict(dummy_input, verbose=0)
        print(f"   Keras model prediction: {keras_pred[0][0]:.4f}")
        
        print("✅ All models saved and verified!")
        
        # Get model size
        h5_size = os.path.getsize(h5_path) / (1024*1024)
        keras_size = os.path.getsize(keras_path) / (1024*1024)
        print(f"\n📊 Model sizes:")
        print(f"   H5 file: {h5_size:.1f} MB")
        print(f"   Keras file: {keras_size:.1f} MB")
        
    else:
        print(f"❌ No trained weights found at: {weights_path}")
        print("Please run training first!")

if __name__ == "__main__":
    main()