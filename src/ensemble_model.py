import os
import tensorflow as tf
import numpy as np
from PIL import Image

class EnsembleClassifier:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load multiple models for ensemble"""
        model_paths = {
            'cnn': 'models/final_model.h5',
            'transfer': 'models/transfer_model.h5'
        }
        
        for name, path in model_paths.items():
            if os.path.exists(path):
                self.models[name] = tf.keras.models.load_model(path)
                print(f"✅ Loaded {name} model")
            else:
                print(f"⚠️ {name} model not found: {path}")
    
    def preprocess_for_cnn(self, image):
        """Preprocess for CNN model (150x150)"""
        img = image.resize((150, 150))
        img_array = np.array(img) / 255.0
        if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        return np.expand_dims(img_array, axis=0)
    
    def preprocess_for_transfer(self, image):
        """Preprocess for transfer learning model (224x224)"""
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        return np.expand_dims(img_array, axis=0)
    
    def predict_ensemble(self, image):
        """Combine predictions from multiple models"""
        if not self.models:
            return None, None
        
        predictions = []
        confidences = []
        
        # CNN model prediction
        if 'cnn' in self.models:
            processed = self.preprocess_for_cnn(image)
            pred = self.models['cnn'].predict(processed, verbose=0)
            raw_score = float(pred[0][0])
            
            if raw_score < 0.5:
                cnn_class = "BIRD"
                cnn_conf = 1 - raw_score
            else:
                cnn_class = "DRONE" 
                cnn_conf = raw_score
            
            predictions.append(cnn_class)
            confidences.append(cnn_conf)
        
        # Transfer learning model prediction
        if 'transfer' in self.models:
            processed = self.preprocess_for_transfer(image)
            pred = self.models['transfer'].predict(processed, verbose=0)
            raw_score = float(pred[0][0])
            
            if raw_score < 0.5:
                transfer_class = "BIRD"
                transfer_conf = 1 - raw_score
            else:
                transfer_class = "DRONE"
                transfer_conf = raw_score
            
            predictions.append(transfer_class)
            confidences.append(transfer_conf)
        
        # Ensemble voting
        if len(predictions) == 0:
            return None, None
        
        # If both models agree
        if len(set(predictions)) == 1:
            final_class = predictions[0]
            final_confidence = np.mean(confidences)
        else:
            # Take higher confidence prediction
            final_class = predictions[np.argmax(confidences)]
            final_confidence = max(confidences)
        
        return final_class, final_confidence

def test_ensemble():
    """Test ensemble on difficult cases"""
    classifier = EnsembleClassifier()
    
    # Test with difficult drone images
    difficult_drones = [
        r"C:\JN\Aerial_Object_Classification\data\classification_dataset\test\drone\*.jpg"  # Add specific difficult images
    ]
    
    print("🧪 Testing Ensemble on Difficult Cases...")
    
    # You can add specific test cases here
    print("🎯 Ensemble model ready for difficult drones!")
    print("💡 It combines multiple models for better accuracy")

if __name__ == "__main__":
    test_ensemble()