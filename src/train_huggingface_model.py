import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

print("🚀 Training Hugging Face Compatible Model...")

def create_compatible_model():
    """Create a model compatible with latest TensorFlow"""
    
    # Simple CNN compatible with all TF versions
    model = keras.Sequential([
        # Input layer (compatible with all versions)
        keras.layers.Input(shape=(150, 150, 3)),
        
        # Conv layers
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        
        # Classifier
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_data_generators(data_path):
    """Create data generators"""
    
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=(150, 150),
        batch_size=16,
        class_mode='binary',
        classes=['bird', 'drone'],
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_path, 'valid'),
        target_size=(150, 150),
        batch_size=16,
        class_mode='binary',
        classes=['bird', 'drone'],
        shuffle=True
    )
    
    return train_gen, val_gen

def main():
    print(f"TensorFlow version: {tf.__version__}")
    
    data_path = r"C:\JN\Aerial_Object_Classification\data\classification_dataset"
    
    # Create and train model
    model = create_compatible_model()
    print("\n📋 Model Summary:")
    model.summary()
    
    train_gen, val_gen = create_data_generators(data_path)
    
    print(f"\n🎯 Training on {train_gen.samples} images...")
    
    # Quick training (3 epochs for testing)
    history = model.fit(
        train_gen,
        epochs=3,
        validation_data=val_gen,
        verbose=1
    )
    
    # Save in compatible format
    os.makedirs('../models', exist_ok=True)
    
    # Save in multiple formats
    model.save('../models/hf_model.h5')  # H5 format
    model.save('../models/hf_model.keras')  # New keras format
    
    print("\n✅ Models saved:")
    print("   - models/hf_model.h5")
    print("   - models/hf_model.keras")
    
    # Test the model
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(data_path, 'test'),
        target_size=(150, 150),
        batch_size=16,
        class_mode='binary',
        classes=['bird', 'drone'],
        shuffle=False
    )
    
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
    print(f"\n🎉 Test Accuracy: {test_accuracy:.2%}")
    
    # Quick test with sample images
    print("\n🧪 Quick Test Predictions:")
    
    # Test one bird image
    bird_test_path = os.path.join(data_path, 'test', 'bird')
    if os.path.exists(bird_test_path):
        bird_images = [f for f in os.listdir(bird_test_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if bird_images:
            img_path = os.path.join(bird_test_path, bird_images[0])
            try:
                from PIL import Image
                img = Image.open(img_path)
                img = img.resize((150, 150))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                prediction = model.predict(img_array, verbose=0)
                score = float(prediction[0][0])
                
                if score < 0.5:
                    predicted = "BIRD"
                    confidence = (1 - score) * 100
                else:
                    predicted = "DRONE"
                    confidence = score * 100
                
                print(f"   Bird image test: {predicted} ({confidence:.1f}%)")
            except Exception as e:
                print(f"   Bird test error: {e}")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")