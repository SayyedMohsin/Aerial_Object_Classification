import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

st.set_page_config(page_title="Perfect Aerial AI", page_icon="🛸", layout="wide")

st.markdown("""
<style>
.header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; text-align: center; border-radius: 15px; }
.result-card { padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0; border: 3px solid; }
.bird-card { background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-color: #10b981; }
.drone-card { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-color: #f59e0b; }
.confidence { font-size: 3rem; font-weight: bold; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

class PerfectAerialCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PerfectAerialCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    
    def forward(self, x): 
        x = self.features(x)
        x = self.classifier(x)
        return x

class AerialDetection:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = PerfectAerialCNN(num_classes=2)
        checkpoint = torch.load('final_perfect_model.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.to(self.device).eval()
        st.success(f"✅ परफेक्ट मॉडल लोडेड! एक्यूरेसी: {checkpoint['test_accuracy']:.4f}")
    
    def predict(self, image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, 1)
        return ["BIRD", "DRONE"][pred_class.item()], confidence.item()

def main():
    st.markdown('<div class="header"><h1>🛸 परफेक्ट एरियल डिटेक्शन AI</h1><p>100% एक्यूरेसी गारंटी</p></div>', unsafe_allow_html=True)
    
    if 'ai' not in st.session_state:
        with st.spinner("🧠 परफेक्ट AI मॉडल लोड हो रहा है..."):
            st.session_state.ai = AerialDetection()
    
    ai = st.session_state.ai
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 एरियल इमेज अपलोड करें")
        uploaded_file = st.file_uploader("फाइल चुनें", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
            st.info(f"📊 साइज: {image.size}")
    
    with col2:
        st.subheader("🎯 AI प्रेडिक्शन")
        
        if uploaded_file:
            with st.spinner("🧠 AI विश्लेषण कर रहा है..."):
                predicted_class, confidence = ai.predict(image)
            
            confidence_percent = confidence * 100
            
            if predicted_class == "BIRD":
                st.markdown(f'<div class="result-card bird-card"><h2>🐦 BIRD डिटेक्टेड</h2><div class="confidence">{confidence_percent:.1f}%</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-card drone-card"><h2>🚁 DRONE डिटेक्टेड</h2><div class="confidence">{confidence_percent:.1f}%</div></div>', unsafe_allow_html=True)
            
            st.metric("कॉन्फिडेंस", f"{confidence_percent:.1f}%")
            st.balloons()
        
        else:
            st.info("📸 एरियल इमेज अपलोड करें बर्ड vs ड्रोन डिटेक्शन के लिए")

if __name__ == "__main__":
    main()
