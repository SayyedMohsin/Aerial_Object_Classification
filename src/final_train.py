import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import cv2

print("🔥 FINAL TRAINING FOR PERFECT PREDICTIONS...")

def preprocess_image_cv2(image_path, target_size=(224, 224)):
    """Professional preprocessing like real deployment"""
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Normalize
    img = img / 255.0
    
    return img

def create_final_model():
    """Create optimized model"""
    model = keras.Sequential([
        # Enhanced architecture
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.3),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),
        
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.5),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model

def test_real_images(model, data_path):
    """Test on actual images"""
    print("\n🧪 TESTING ON REAL IMAGES:")
    
    # Test bird images
    bird_path = os.path.join(data_path, 'test', 'bird')
    bird_images = [f for f in os.listdir(bird_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
    
    correct_birds = 0
    for img_name in bird_images:
        img_path = os.path.join(bird_path, img_name)
        img = preprocess_image_cv2(img_path)
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
        score = float(pred[0][0])
        
        if score < 0.5:
            correct_birds += 1
            print(f"   ✅ {img_name}: BIRD (score: {score:.4f})")
        else:
            print(f"   ❌ {img_name}: WRONG as DRONE (score: {score:.4f})")
    
    # Test drone images
    drone_path = os.path.join(data_path, 'test', 'drone')
    drone_images = [f for f in os.listdir(drone_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
    
    correct_drones = 0
    for img_name in drone_images:
        img_path = os.path.join(drone_path, img_name)
        img = preprocess_image_cv2(img_path)
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
        score = float(pred[0][0])
        
        if score > 0.5:
            correct_drones += 1
            print(f"   ✅ {img_name}: DRONE (score: {score:.4f})")
        else:
            print(f"   ❌ {img_name}: WRONG as BIRD (score: {score:.4f})")
    
    print(f"\n📊 Real Test Accuracy:")
    print(f"   Birds: {correct_birds}/5 correct")
    print(f"   Drones: {correct_drones}/5 correct")
    print(f"   Total: {(correct_birds + correct_drones)}/10 correct")

def main():
    data_path = r"C:\JN\Aerial_Object_Classification\data\classification_dataset"
    
    print(f"📁 Data path: {data_path}")
    
    # Data generators with heavy augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
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
    
    print(f"🎯 Training on {train_gen.samples} images")
    print(f"🎯 Validating on {val_gen.samples} images")
    
    # Create and train model
    model = create_final_model()
    print("\n📋 Model Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=15, 
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=8, 
            factor=0.5,
            min_lr=0.00001
        ),
        keras.callbacks.ModelCheckpoint(
            '../models/best_final_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    print("\n🔥 Training for 30 epochs (Be patient for PERFECT results)...")
    history = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    os.makedirs('../models', exist_ok=True)
    model.save('../models/final_perfect_model.h5')
    print("\n✅ Model saved: models/final_perfect_model.h5")
    
    # Test on validation
    val_loss, val_acc, val_precision, val_recall = model.evaluate(val_gen, verbose=0)
    print(f"\n🎯 Validation Results:")
    print(f"   Accuracy:  {val_acc:.2%}")
    print(f"   Precision: {val_precision:.2%}")
    print(f"   Recall:    {val_recall:.2%}")
    
    # Test on real images
    test_real_images(model, data_path)
    
    print("\n🎉 TRAINING COMPLETE! Use 'final_perfect_model.h5'")

if __name__ == "__main__":
    main()