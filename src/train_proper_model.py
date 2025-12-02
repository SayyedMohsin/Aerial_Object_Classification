import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("🔥 TRAINING PROPER MODEL FOR PERFECT PREDICTIONS...")

def create_proper_model():
    """Create a robust model that actually learns"""
    
    model = keras.Sequential([
        # Input - NO Input layer, specify in Conv2D
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
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def main():
    data_path = r"C:\JN\Aerial_Object_Classification\data\classification_dataset"
    
    print(f"📁 Using data from: {data_path}")
    
    # Data generators
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=(224, 224),
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
    
    print(f"\n🎯 Class mapping: {train_gen.class_indices}")
    print(f"   This means: 0 = BIRD, 1 = DRONE")
    
    # Create and train model
    model = create_proper_model()
    print("\n📋 Model Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    
    print("\n🔥 Training for 20 epochs (this will take time)...")
    history = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model in SIMPLE format
    os.makedirs('../models', exist_ok=True)
    
    # Save weights ONLY (not full model)
    model.save_weights('../models/proper_model_weights.weights.h5')
    
    # Save model architecture separately
    model_json = model.to_json()
    with open('../models/model_architecture.json', 'w') as f:
        f.write(model_json)
    
    print("\n✅ Model saved:")
    print("   - proper_model_weights.h5 (weights only)")
    print("   - model_architecture.json (architecture)")
    
    # Test accuracy
    test_gen = val_datagen.flow_from_directory(
        os.path.join(data_path, 'test'),
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        classes=['bird', 'drone'],
        shuffle=False
    )
    
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_gen, verbose=0)
    print(f"\n🎉 FINAL TEST RESULTS:")
    print(f"   Accuracy:  {test_acc:.2%}")
    print(f"   Precision: {test_precision:.2%}")
    print(f"   Recall:    {test_recall:.2%}")
    
    # Test predictions
    print("\n🧪 Sample Predictions:")
    test_images, test_labels = next(test_gen)
    
    for i in range(3):
        pred = model.predict(np.expand_dims(test_images[i], axis=0), verbose=0)
        true_class = "BIRD" if test_labels[i] == 0 else "DRONE"
        pred_class = "BIRD" if pred[0][0] < 0.5 else "DRONE"
        
        print(f"   Sample {i+1}: True={true_class}, Predicted={pred_class}, Score={pred[0][0]:.4f}")
        print(f"   Status: {'✅ CORRECT' if true_class == pred_class else '❌ WRONG'}")

if __name__ == "__main__":
    main()