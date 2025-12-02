import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

print("🚀 STARTING FINAL CORRECT TRAINING...")

class FinalTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.history = None
        
    def create_data_generators(self):
        print("📂 Creating Data Generators with CORRECT labels...")
        
        # Data augmentation
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            horizontal_flip=True,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2
        )
        
        val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Create generators - VERIFY CLASS LABELS
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_path, 'train'),
            target_size=(150, 150),
            batch_size=16,
            class_mode='binary',
            classes=['bird', 'drone'],  # This order is IMPORTANT
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            os.path.join(self.data_path, 'valid'),
            target_size=(150, 150),
            batch_size=16,
            class_mode='binary',
            classes=['bird', 'drone'],
            shuffle=True
        )
        
        test_generator = val_datagen.flow_from_directory(
            os.path.join(self.data_path, 'test'),
            target_size=(150, 150),
            batch_size=16,
            class_mode='binary',
            classes=['bird', 'drone'],
            shuffle=False
        )
        
        print(f"🎯 CLASS MAPPING: {train_generator.class_indices}")
        print(f"   This should be: {{'bird': 0, 'drone': 1}}")
        print(f"📊 Samples - Train: {train_generator.samples}, Val: {validation_generator.samples}")
        
        return train_generator, validation_generator, test_generator
    
    def build_improved_model(self):
        print("🏗️ Building Improved CNN Model...")
        
        model = keras.Sequential([
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_model(self):
        print("🎯 Training Model with CORRECT Labels...")
        
        train_gen, val_gen, test_gen = self.create_data_generators()
        
        # Build model
        self.model = self.build_improved_model()
        
        print("\n📋 Model Summary:")
        self.model.summary()
        
        # Create models directory
        os.makedirs('../models', exist_ok=True)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
            keras.callbacks.ModelCheckpoint(
                '../models/final_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
        ]
        
        print("\n🔥 Training for-10 epochs...")
        self.history = self.model.fit(
            train_gen,
            epochs=10,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Final evaluation
        print("\n🧪 Final Evaluation...")
        test_results = self.model.evaluate(test_gen, verbose=0)
        
        print(f"\n🎉 FINAL RESULTS:")
        print(f"   Test Loss: {test_results[0]:.4f}")
        print(f"   Test Accuracy: {test_results[1]:.4f}")
        print(f"   Test Precision: {test_results[2]:.4f}")
        print(f"   Test Recall: {test_results[3]:.4f}")
        
        return test_gen
    
    def verify_predictions(self, test_gen):
        print("\n🔍 Verifying Predictions...")
        
        # Test few samples manually
        test_images, test_labels = test_gen.next()
        
        print(f"Testing {len(test_images)} samples...")
        for i in range(min(3, len(test_images))):
            img = test_images[i]
            true_label = "Bird" if test_labels[i] == 0 else "Drone"
            
            prediction = self.model.predict(np.expand_dims(img, axis=0), verbose=0)
            pred_score = float(prediction[0][0])
            pred_label = "Bird" if pred_score > 0.5 else "Drone"
            
            print(f"   Sample {i+1}: True={true_label}, Predicted={pred_label}, Score={pred_score:.4f}")
            print(f"   Status: {'✅ CORRECT' if true_label == pred_label else '❌ WRONG'}")

def main():
    data_path = r"C:\JN\Aerial_Object_Classification\data\classification_dataset"
    
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        return
    
    trainer = FinalTrainer(data_path)
    
    try:
        test_gen = trainer.train_model()
        trainer.verify_predictions(test_gen)
        
        print("\n🎊 TRAINING COMPLETED SUCCESSFULLY!")
        print("💾 Model saved as: models/final_model.h5")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()