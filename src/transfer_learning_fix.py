import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
import numpy as np

print("🚀 APPLYING TRANSFER LEARNING FIX...")

def create_transfer_model():
    """Use pre-trained MobileNetV2 for better feature extraction"""
    
    # Load pre-trained model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classifier
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    data_path = r"C:\JN\Aerial_Object_Classification\data\classification_dataset"
    
    # Data generators with larger image size
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        horizontal_flip=True,
        zoom_range=0.3,
        brightness_range=[0.8, 1.2]
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=(224, 224),  # Larger size for better details
        batch_size=16,
        class_mode='binary',
        classes=['bird', 'drone'],
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_path, 'valid'),
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        classes=['bird', 'drone'],
        shuffle=True
    )
    
    print(f"🎯 Training with {train_gen.samples} images...")
    
    # Create and train model
    model = create_transfer_model()
    print("\n📋 Transfer Learning Model Summary:")
    model.summary()
    
    # Train
    print("\n🔥 Training Transfer Learning Model...")
    history = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        verbose=1
    )
    
    # Save model
    os.makedirs('../models', exist_ok=True)
    model.save('../models/transfer_model.h5')
    print("✅ Model saved: models/transfer_model.h5")
    
    # Evaluate
    test_gen = val_datagen.flow_from_directory(
        os.path.join(data_path, 'test'),
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        classes=['bird', 'drone'],
        shuffle=False
    )
    
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
    print(f"🎉 Transfer Learning Test Accuracy: {test_accuracy:.2%}")
    
    print("\n🎊 TRANSFER LEARNING COMPLETED!")
    print("💡 This should handle small drones better!")

if __name__ == "__main__":
    main()