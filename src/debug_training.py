import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def debug_data_loading():
    print("🔍 DEBUGGING DATA LOADING...")
    
    data_path = r"C:\JN\Aerial_Object_Classification\data\classification_dataset"
    
    # Check data generators
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_gen = datagen.flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        classes=['bird', 'drone'],
        shuffle=True
    )
    
    print(f"✅ Classes found: {train_gen.class_indices}")
    print(f"✅ Number of samples: {train_gen.samples}")
    
    # Get one batch to check
    batch_x, batch_y = train_gen.next()
    print(f"✅ Batch shape: {batch_x.shape}")
    print(f"✅ Batch labels: {batch_y[:10]}")  # First 10 labels
    
    # Check if labels are correct (0 for bird, 1 for drone)
    unique_labels = np.unique(batch_y)
    print(f"✅ Unique labels in batch: {unique_labels}")
    
    return train_gen

def build_better_model():
    print("\n🏗️ BUILDING BETTER MODEL...")
    
    model = keras.Sequential([
        # First Conv Block
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        keras.layers.MaxPooling2D(2, 2),
        
        # Second Conv Block  
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        
        # Third Conv Block
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        
        # Flatten and Dense
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Use different optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def quick_train():
    print("\n🎯 QUICK TRAINING (3 EPOCHS ONLY)...")
    
    data_path = r"C:\JN\Aerial_Object_Classification\data\classification_dataset"
    
    # Simple data generator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_gen = datagen.flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=(224, 224),
        batch_size=16,  # Smaller batch
        class_mode='binary',
        classes=['bird', 'drone']
    )
    
    val_gen = datagen.flow_from_directory(
        os.path.join(data_path, 'valid'), 
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        classes=['bird', 'drone']
    )
    
    # Build model
    model = build_better_model()
    model.summary()
    
    # Train for only 3 epochs to test
    history = model.fit(
        train_gen,
        epochs=3,
        validation_data=val_gen,
        verbose=1
    )
    
    # Quick evaluation
    test_gen = datagen.flow_from_directory(
        os.path.join(data_path, 'test'),
        target_size=(224, 224), 
        batch_size=16,
        class_mode='binary',
        classes=['bird', 'drone'],
        shuffle=False
    )
    
    print("\n🧪 QUICK EVALUATION:")
    results = model.evaluate(test_gen, verbose=0)
    
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}") 
    print(f"Test Recall: {results[3]:.4f}")
    
    # Check if results are reasonable
    if results[1] > 0.6:  # If accuracy > 60%
        print("\n✅ MODEL IS WORKING! Now train with more epochs.")
        return True
    else:
        print("\n❌ MODEL STILL HAS ISSUES! Need to debug further.")
        return False

if __name__ == "__main__":
    print("🚀 STARTING DEBUGGING...")
    
    # Step 1: Debug data loading
    debug_data_loading()
    
    # Step 2: Quick training test
    success = quick_train()
    
    if success:
        print("\n🎉 Debugging successful! Now run the main training script again.")
    else:
        print("\n🔧 More debugging needed. Check data paths and preprocessing.")
    
    input("\nPress Enter to exit...")