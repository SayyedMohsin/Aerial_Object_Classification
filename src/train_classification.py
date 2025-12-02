import os
import sys

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print("✅ TensorFlow imported successfully!")
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow import error: {e}")
    print("Please install TensorFlow using: pip install tensorflow")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    print("✅ Other packages imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class ClassificationTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.history = None
        
    def create_data_generators(self):
        print("📂 Creating Data Generators...")
        
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2]
        )
        
        # Only rescaling for validation and test
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_path, 'train'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            classes=['bird', 'drone'],
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            os.path.join(self.data_path, 'valid'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            classes=['bird', 'drone'],
            shuffle=True
        )
        
        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.data_path, 'test'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            classes=['bird', 'drone'],
            shuffle=False
        )
        
        print(f"✅ Training samples: {train_generator.samples}")
        print(f"✅ Validation samples: {validation_generator.samples}")
        print(f"✅ Test samples: {test_generator.samples}")
        
        return train_generator, validation_generator, test_generator
    
    def build_simple_cnn(self):
        print("🏗️ Building Simple CNN Model...")
        
        model = keras.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_model(self):
        print("🎯 Starting Model Training...")
        
        # Get data generators
        train_gen, val_gen, test_gen = self.create_data_generators()
        
        # Build model
        self.model = self.build_simple_cnn()
        
        print("\n📋 Model Summary:")
        self.model.summary()
        
        # Create models directory if it doesn't exist
        os.makedirs('../models', exist_ok=True)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ModelCheckpoint(
                '../models/custom_cnn_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                patience=3, 
                factor=0.5,
                verbose=1
            )
        ]
        
        print("\n🔥 Training Started...")
        # Train model with fewer epochs for testing
        self.history = self.model.fit(
            train_gen,
            epochs=10,  # Reduced for testing
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\n🧪 Evaluating on Test Set...")
        test_results = self.model.evaluate(test_gen, verbose=0)
        
        print(f"\n📊 Test Results:")
        print(f"   Loss: {test_results[0]:.4f}")
        print(f"   Accuracy: {test_results[1]:.4f}")
        print(f"   Precision: {test_results[2]:.4f}")
        print(f"   Recall: {test_results[3]:.4f}")
        
        return test_gen
    
    def plot_training_history(self):
        print("📈 Plotting Training History...")
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot accuracy
            axes[0].plot(self.history.history['accuracy'], label='Training Accuracy', marker='o')
            axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', marker='o')
            axes[0].set_title('Model Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True)
            
            # Plot loss
            axes[1].plot(self.history.history['loss'], label='Training Loss', marker='o')
            axes[1].plot(self.history.history['val_loss'], label='Validation Loss', marker='o')
            axes[1].set_title('Model Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig('../models/training_history.png', dpi=300, bbox_inches='tight')
            print("✅ Training history saved as 'training_history.png'")
            plt.show()
            
        except Exception as e:
            print(f"❌ Error plotting training history: {e}")

def main():
    data_path = r"C:\JN\Aerial_Object_Classification\data\classification_dataset"
    
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        return
    
    print("🚀 Starting Classification Model Training...")
    print(f"Data path: {data_path}")
    
    trainer = ClassificationTrainer(data_path)
    
    try:
        test_gen = trainer.train_model()
        trainer.plot_training_history()
        
        print("\n✅ Model training completed successfully!")
        print("💾 Model saved as: models/custom_cnn_model.h5")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("This might be due to:")
        print("1. TensorFlow installation issues")
        print("2. GPU memory issues")
        print("3. Data path problems")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()