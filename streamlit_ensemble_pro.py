import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# Add current directory to path to import ensemble_model
sys.path.append(os.path.dirname(__file__))

try:
    from ensemble_model import EnsembleClassifier
    IMPORT_SUCCESS = True
except ImportError as e:
    st.error(f"❌ Could not import EnsembleClassifier: {e}")
    IMPORT_SUCCESS = False

st.set_page_config(
    page_title="Aerial AI Classification - Professional",
    page_icon="🤖", 
    layout="wide"
)

# 1. CUSTOM PROFESSIONAL CSS INJECTION
st.markdown("""
<style>
/* Main Professional Styles */
.main-header {
    text-align: center;
    color: #1e40af;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.sub-header {
    text-align: center;
    color: #6b7280;
    font-size: 1.3rem;
    margin-bottom: 3rem;
    font-weight: 300;
}

/* Professional button style */
.professional-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 30px;
    border-radius: 12px;
    font-weight: 600;
    cursor: pointer;
    text-align: center;
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    transition: all 0.3s ease;
    border: none;
    font-size: 1rem;
    margin: 10px 0;
    display: inline-block;
    text-decoration: none;
}
.professional-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(102, 126, 234, 0.4);
    color: white;
    text-decoration: none;
}

/* Model cards with professional design */
.model-card {
    padding: 1.5rem;
    border-radius: 16px;
    margin: 1rem 0;
    border-left: 6px solid;
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
    transition: transform 0.3s ease;
    background: white;
}
.model-card:hover {
    transform: translateY(-3px);
}
.cnn-card {
    border-left-color: #10b981;
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
}
.transfer-card {
    border-left-color: #3b82f6;
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
}
.ensemble-card {
    border-left-color: #f59e0b;
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
}

/* Confidence indicators */
.confidence-high { 
    color: #10b981; 
    font-weight: 800; 
    font-size: 2.5rem;
    text-shadow: 2px 2px 4px rgba(16, 185, 129, 0.2);
}
.confidence-medium { 
    color: #f59e0b; 
    font-weight: 800; 
    font-size: 2.5rem;
    text-shadow: 2px 2px 4px rgba(245, 158, 11, 0.2);
}
.confidence-low { 
    color: #ef4444; 
    font-weight: 800; 
    font-size: 2.5rem;
    text-shadow: 2px 2px 4px rgba(239, 68, 68, 0.2);
}

/* Upload area styling */
.upload-area {
    border: 3px dashed #3b82f6;
    border-radius: 20px;
    padding: 3rem;
    text-align: center;
    margin: 2rem 0;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    transition: all 0.3s ease;
}
.upload-area:hover {
    border-color: #1d4ed8;
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
}

/* Prediction result cards */
.prediction-card {
    padding: 2.5rem;
    border-radius: 20px;
    margin: 2rem 0;
    text-align: center;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
    border: 3px solid;
    transition: all 0.4s ease;
    background: white;
}
.prediction-card:hover {
    transform: scale(1.02);
}
.bird-prediction {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    border-color: #10b981;
}
.drone-prediction {
    background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
    border-color: #ef4444;
}

/* Sidebar professional styling */
.sidebar-content {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
}

/* Progress bar styling */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #f1f5f9;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

def main():
    # PROFESSIONAL HEADER SECTION
    st.markdown('<h1 class="main-header">🤖 एरियल एआई क्लासिफिकेशन</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">प्रोफेशनल बर्ड vs ड्रोन डिटेक्शन सिस्टम</p>', unsafe_allow_html=True)
    
    if not IMPORT_SUCCESS:
        st.error("""
        ❌ एन्सेम्बल मॉड्यूल नहीं मिला!
        कृपया सुनिश्चित करें कि `ensemble_model.py` प्रोजेक्ट डायरेक्टरी में मौजूद है।
        """)
        return
    
    # Initialize classifier
    classifier = EnsembleClassifier()
    
    # PROFESSIONAL SIDEBAR
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3 style="color: #1e40af; margin-bottom: 2rem;">🔧 सिस्टम स्टेटस</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if classifier.models:
            st.markdown(f"""
            <div class="metric-card">
                <h4>✅ मॉडल लोडेड</h4>
                <h2>{len(classifier.models)}/2</h2>
            </div>
            """, unsafe_allow_html=True)
            
            for model_name in classifier.models.keys():
                accuracy = "83.26%" if model_name == 'cnn' else "98.14%"
                st.markdown(f"""
                <div class="sidebar-content">
                    <h5>{"🧠 सीएनएन मॉडल" if model_name == 'cnn' else "🚀 ट्रांसफर लर्निंग"}</h5>
                    <p><strong>एक्यूरेसी:</strong> {accuracy}</p>
                    <p><strong>इनपुट:</strong> {"150×150" if model_name == 'cnn' else "224×224"}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ कोई मॉडल लोड नहीं हुआ")
            st.info("""
            आवश्यक मॉडल:
            - models/final_model.h5 (CNN)
            - models/transfer_model.h5 (Transfer Learning)
            """)
        
        # Quick action buttons in sidebar
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <a href="#" class="professional-button">📊 डैशबोर्ड</a>
            <a href="#" class="professional-button" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">🔄 रीसेट</a>
        </div>
        """, unsafe_allow_html=True)

    # MAIN CONTENT AREA
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%); 
                    padding: 2rem; border-radius: 20px; border-left: 6px solid #00bcd4;">
            <h4 style="font-weight: bold; color: #00838f; margin-bottom: 1rem;">📤 इमेज अपलोड</h4>
            <p style="color: #455a64; margin-bottom: 1.5rem;">कठिन ड्रोन/बर्ड इमेज अपलोड करें जो पहले गलत क्लासिफाई हुई थी</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Professional file uploader
        uploaded_file = st.file_uploader(
            "इमेज फाइल चुनें", 
            type=['jpg', 'jpeg', 'png'],
            help="छोटे ड्रोन, कॉम्प्लेक्स बैकग्राउंड वाली इमेज अपलोड करें"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"इमेज साइज: {image.size}", use_column_width=True)
                
                # Image analysis in professional card
                st.markdown("""
                <div class="model-card" style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); border-left-color: #8e24aa;">
                    <h5 style="color: #6a1b9a; margin-bottom: 1rem;">📷 इमेज एनालिसिस</h5>
                    <p style="color: #4a148c; margin: 0.2rem 0;"><strong>डाइमेंशन:</strong> {}</p>
                    <p style="color: #4a148c; margin: 0.2rem 0;"><strong>फॉर्मेट:</strong> {}</p>
                    <p style="color: #4a148c; margin: 0.2rem 0;"><strong>मोड:</strong> {}</p>
                </div>
                """.format(image.size, image.format, image.mode), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ इमेज लोडिंग एरर: {e}")
                return
    
    with col2:
        if uploaded_file is not None:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                        padding: 2rem; border-radius: 20px; border-left: 6px solid #ff9800;">
                <h4 style="font-weight: bold; color: #e65100; margin-bottom: 1rem;">🎯 प्रिडिक्शन रिजल्ट</h4>
                <p style="color: #bf360c;">एन्सेम्बल एआई मॉडल विश्लेषण कर रहा है...</p>
            </div>
            """, unsafe_allow_html=True)
            
            if not classifier.models:
                st.error("प्रिडिक्शन के लिए कोई मॉडल उपलब्ध नहीं!")
                return
            
            # Make prediction with professional loading
            with st.spinner("🤖 मल्टीपल एआई मॉडल एनालाइज कर रहे हैं..."):
                class_name, confidence, raw_scores = classifier.predict_ensemble(image)
            
            if class_name:
                confidence_percent = confidence * 100
                
                # Professional confidence level
                if confidence_percent >= 85:
                    conf_class = "confidence-high"
                    conf_text = "बहुत उच्च आत्मविश्वास"
                    conf_emoji = "🟢"
                elif confidence_percent >= 70:
                    conf_class = "confidence-medium" 
                    conf_text = "उच्च आत्मविश्वास"
                    conf_emoji = "🟡"
                elif confidence_percent >= 60:
                    conf_class = "confidence-medium"
                    conf_text = "अच्छा आत्मविश्वास"
                    conf_emoji = "🟡"
                else:
                    conf_class = "confidence-low"
                    conf_text = "कम आत्मविश्वास"
                    conf_emoji = "🔴"
                
                # Professional prediction display
                if class_name == "BIRD":
                    st.markdown(f"""
                    <div class="prediction-card bird-prediction">
                        <h1 style="font-size: 3rem; margin-bottom: 1rem;">🐦 बर्ड डिटेक्टेड</h1>
                        <div class="{conf_class}">{conf_emoji} {confidence_percent:.2f}%</div>
                        <h3 style="color: #065f46; margin-top: 1rem;">{conf_text}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="model-card cnn-card">
                        <h5 style="color: #065f46;">🐦 बर्ड कैरेक्टरिस्टिक्स</h5>
                        <ul style="color: #047857;">
                        <li>ऑर्गेनिक, कर्व्ड शेप्स और कॉन्टूर्स</li>
                        <li>फेदर टेक्सचर और पैटर्न विजिबल</li>
                        <li>नेचुरल विंग फॉर्मेशन</li>
                        <li>स्मूथ, फ्लोइंग बॉडी लाइन्स</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown(f"""
                    <div class="prediction-card drone-prediction">
                        <h1 style="font-size: 3rem; margin-bottom: 1rem;">🚁 ड्रोन डिटेक्टेड</h1>
                        <div class="{conf_class}">{conf_emoji} {confidence_percent:.2f}%</div>
                        <h3 style="color: #7f1d1d; margin-top: 1rem;">{conf_text}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="model-card transfer-card">
                        <h5 style="color: #1e40af;">🚁 ड्रोन कैरेक्टरिस्टिक्स</h5>
                        <ul style="color: #3730a3;">
                        <li>मैकेनिकल, एंगुलर स्ट्रक्चर</li>
                        <li>विजिबल प्रोपेलर्स (आमतौर पर 4 या अधिक)</li>
                        <li>स्ट्रेट एजेस और ज्योमेट्रिक शेप्स</li>
                        <li>मैन-मेड कंपोनेंट्स और मटेरियल्स</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Professional model breakdown
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                            padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
                    <h5 style="color: #475569; margin-bottom: 1rem;">🔍 मॉडल ब्रेकडाउन</h5>
                </div>
                """, unsafe_allow_html=True)
                
                if 'cnn' in classifier.models:
                    st.markdown("""
                    <div class="model-card cnn-card">
                        <h6 style="color: #065f46; margin-bottom: 0.5rem;">🧠 सीएनएन मॉडल</h6>
                        <p style="color: #047857; margin: 0; font-size: 0.9rem;">इनपुट: 150×150 | एक्यूरेसी: 83.26%</p>
                        <p style="color: #047857; margin: 0; font-size: 0.9rem;">बेस्ट फॉर: स्टैंडर्ड केसेस, क्लियर इमेजेस</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if 'transfer' in classifier.models:
                    st.markdown("""
                    <div class="model-card transfer-card">
                        <h6 style="color: #1e40af; margin-bottom: 0.5rem;">🚀 ट्रांसफर लर्निंग मॉडल</h6>
                        <p style="color: #3730a3; margin: 0; font-size: 0.9rem;">इनपुट: 224×224 | एक्यूरेसी: 98.14%</p>
                        <p style="color: #3730a3; margin: 0; font-size: 0.9rem;">बेस्ट फॉर: स्मॉल ऑब्जेक्ट्स, कॉम्प्लेक्स बैकग्राउंड</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Raw scores in professional format
                if raw_scores:
                    st.markdown("""
                    <div class="model-card" style="background: linear-gradient(135deg, #fef7cd 0%, #fde68a 100%); border-left-color: #d97706;">
                        <h6 style="color: #92400e; margin-bottom: 0.5rem;">📊 रॉ मॉडल स्कोर</h6>
                    </div>
                    """, unsafe_allow_html=True)
                    for score in raw_scores:
                        st.write(f"`{score}`")
                
                # Special handling for difficult cases
                if confidence_percent < 70:
                    st.markdown("""
                    <div class="model-card" style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-left-color: #ef4444;">
                        <h6 style="color: #dc2626;">⚠️ चुनौतीपूर्ण केस डिटेक्टेड</h6>
                        <p style="color: #b91c1c; margin: 0.5rem 0; font-size: 0.9rem;">
                        इस इमेज में हो सकता है:<br>
                        - बहुत छोटा ड्रोन<br>
                        - कॉम्प्लेक्स बैकग्राउंड (घास, पेड़)<br>
                        - लो रेजोल्यूशन<br>
                        - अनusual एंगल
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Success celebration
                st.balloons()
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0;">
                    <div class="professional-button" style="display: inline-block;">
                        🎉 प्रिडिक्शन सफलतापूर्वक पूरा हुआ!
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.error("❌ प्रिडिक्शन फेल! कृपया दूसरी इमेज ट्राई करें।")

if __name__ == "__main__":
    main()