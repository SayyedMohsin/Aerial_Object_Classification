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
                try:
                    self.models[name] = tf.keras.models.load_model(path)
                    print(f"✅ Loaded {name} model")
                except Exception as e:
                    print(f"❌ Error loading {name} model: {e}")
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
        raw_scores = []
        
        # CNN model prediction
        if 'cnn' in self.models:
            try:
                processed = self.preprocess_for_cnn(image)
                pred = self.models['cnn'].predict(processed, verbose=0)
                raw_score = float(pred[0][0])
                raw_scores.append(f"CNN: {raw_score:.4f}")
                
                if raw_score < 0.5:
                    cnn_class = "BIRD"
                    cnn_conf = 1 - raw_score
                else:
                    cnn_class = "DRONE" 
                    cnn_conf = raw_score
                
                predictions.append(cnn_class)
                confidences.append(cnn_conf)
            except Exception as e:
                print(f"CNN prediction error: {e}")
        
        # Transfer learning model prediction
        if 'transfer' in self.models:
            try:
                processed = self.preprocess_for_transfer(image)
                pred = self.models['transfer'].predict(processed, verbose=0)
                raw_score = float(pred[0][0])
                raw_scores.append(f"Transfer: {raw_score:.4f}")
                
                if raw_score < 0.5:
                    transfer_class = "BIRD"
                    transfer_conf = 1 - raw_score
                else:
                    transfer_class = "DRONE"
                    transfer_conf = raw_score
                
                predictions.append(transfer_class)
                confidences.append(transfer_conf)
            except Exception as e:
                print(f"Transfer prediction error: {e}")
        
        # If no predictions successful
        if len(predictions) == 0:
            return None, None, []
        
        # Ensemble voting
        if len(predictions) == 1:
            # Only one model available
            final_class = predictions[0]
            final_confidence = confidences[0]
        elif len(set(predictions)) == 1:
            # Both models agree
            final_class = predictions[0]
            final_confidence = np.mean(confidences)
        else:
            # Models disagree - take higher confidence
            final_class = predictions[np.argmax(confidences)]
            final_confidence = max(confidences)
        
        return final_class, final_confidence, raw_scores

def test_ensemble():
    """Test ensemble model"""
    print("🧪 Testing Ensemble Classifier...")
    classifier = EnsembleClassifier()
    
    print(f"✅ Loaded {len(classifier.models)} models:")
    for model_name in classifier.models.keys():
        print(f"   - {model_name}")
    
    return classifier

if __name__ == "__main__":
    test_ensemble()
    input("Press Enter to exit...")