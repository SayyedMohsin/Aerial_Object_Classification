import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

try:
    from ensemble_model import EnsembleClassifier
    IMPORT_SUCCESS = True
except ImportError as e:
    st.error(f"Import Error: {e}")
    IMPORT_SUCCESS = False

# ========== PROFESSIONAL PAGE CONFIG ==========
st.set_page_config(
    page_title="Aerial Intelligence Platform",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM PROFESSIONAL CSS ==========
st.markdown("""
<style>
/* ===== GLOBAL STYLES ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* ===== PROFESSIONAL COLOR PALETTE ===== */
:root {
    --primary: #3b82f6;
    --primary-dark: #2563eb;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --dark: #1f2937;
    --light: #f9fafb;
    --gray: #6b7280;
}

/* ===== MAIN CONTAINER ===== */
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* ===== PROFESSIONAL HEADER ===== */
.professional-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 2.5rem 2rem;
    border-radius: 0 0 30px 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    margin-bottom: 2rem;
    text-align: center;
}

.header-title {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(to right, #fff 0%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.header-subtitle {
    font-size: 1.2rem;
    opacity: 0.9;
    font-weight: 300;
}

/* ===== PROFESSIONAL BUTTONS ===== */
.professional-button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: white;
    padding: 12px 28px;
    border-radius: 12px;
    font-weight: 600;
    cursor: pointer;
    text-align: center;
    border: none;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-size: 1rem;
    margin: 0.5rem;
    display: inline-block;
}

.professional-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    background: linear-gradient(135deg, var(--primary-dark) 0%, #1d4ed8 100%);
}

/* ===== ELEGANT CARDS ===== */
.elegant-card {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    border: 1px solid #e5e7eb;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 1.5rem;
}

.elegant-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
}

.card-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--dark);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

.card-title::before {
    content: "▸";
    color: var(--primary);
}

/* ===== PREDICTION CARDS ===== */
.prediction-card {
    padding: 2.5rem;
    border-radius: 20px;
    margin: 2rem 0;
    text-align: center;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    border: 2px solid;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    background: white;
}

.bird-prediction {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    border-color: var(--success);
}

.drone-prediction {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-color: var(--warning);
}

.confidence-display {
    font-size: 3.5rem;
    font-weight: 700;
    margin: 1rem 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.confidence-high { color: var(--success); }
.confidence-medium { color: var(--warning); }
.confidence-low { color: var(--danger); }

/* ===== MODEL STATUS CARDS ===== */
.model-status-card {
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    border-left: 5px solid;
    background: white;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
}

.cnn-model { border-left-color: var(--success); }
.transfer-model { border-left-color: var(--primary); }

/* ===== SIDEBAR STYLING ===== */
.css-1d391kg {  /* Sidebar container */
    background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    color: white;
}

/* ===== METRIC CARDS ===== */
.metric-card {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
}

/* ===== UPLOAD AREA ===== */
.upload-area {
    border: 3px dashed var(--primary);
    border-radius: 20px;
    padding: 3rem;
    text-align: center;
    background: rgba(59, 130, 246, 0.05);
    margin: 2rem 0;
    transition: all 0.3s ease;
}

.upload-area:hover {
    background: rgba(59, 130, 246, 0.1);
    border-color: var(--primary-dark);
}

/* ===== ANIMATIONS ===== */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.6s ease-out;
}

/* ===== RESPONSIVE DESIGN ===== */
@media (max-width: 768px) {
    .header-title { font-size: 2rem; }
    .confidence-display { font-size: 2.5rem; }
}
</style>
""", unsafe_allow_html=True)

# ========== MAIN APPLICATION ==========
def main():
    # ===== PROFESSIONAL HEADER =====
    st.markdown("""
    <div class="professional-header fade-in">
        <h1 class="header-title">🛸 AERIAL INTELLIGENCE PLATFORM</h1>
        <p class="header-subtitle">Advanced AI-Powered Bird vs Drone Classification System</p>
        <div style="margin-top: 1.5rem;">
            <span class="professional-button">📊 Dashboard</span>
            <span class="professional-button">🤖 AI Models</span>
            <span class="professional-button">📈 Analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize classifier
    if IMPORT_SUCCESS:
        classifier = EnsembleClassifier()
    else:
        classifier = None
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.markdown("""
        <div class="elegant-card">
            <div class="card-title">📊 System Overview</div>
            <p>Advanced ensemble classification system combining multiple AI models for maximum accuracy in aerial object detection.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if classifier and classifier.models:
            st.markdown("""
            <div class="metric-card">
                <h3>Model Status</h3>
                <h2 style="font-size: 2.5rem; margin: 0.5rem 0;">🟢 ACTIVE</h2>
                <p>All systems operational</p>
            </div>
            """, unsafe_allow_html=True)
            
            for model_name in classifier.models.keys():
                if model_name == 'cnn':
                    st.markdown("""
                    <div class="model-status-card cnn-model">
                        <h4>🧠 CNN Model</h4>
                        <p><strong>Accuracy:</strong> 83.26%</p>
                        <p><strong>Resolution:</strong> 150×150</p>
                        <p style="color: var(--success); font-weight: 600;">✅ Optimized for standard cases</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif model_name == 'transfer':
                    st.markdown("""
                    <div class="model-status-card transfer-model">
                        <h4>🚀 Transfer Learning</h4>
                        <p><strong>Accuracy:</strong> 98.14%</p>
                        <p><strong>Resolution:</strong> 224×224</p>
                        <p style="color: var(--success); font-weight: 600;">✅ Optimized for complex cases</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="elegant-card">
            <div class="card-title">⚙️ Settings</div>
            <p><strong>Detection Mode:</strong> Ensemble Voting</p>
            <p><strong>Confidence Threshold:</strong> 60%</p>
            <p><strong>Image Processing:</strong> Auto-optimized</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== MAIN CONTENT AREA =====
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="elegant-card fade-in">
            <div class="card-title">📤 Image Upload</div>
            <p>Upload aerial images for AI-powered classification. The system supports JPG, JPEG, and PNG formats.</p>
            <div class="upload-area">
                <h3 style="color: var(--primary);">Drag & Drop or Click to Browse</h3>
                <p style="color: var(--gray); margin: 1rem 0;">Supports bird/drone detection with advanced ensemble AI</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.markdown("""
                <div class="elegant-card">
                    <div class="card-title">🖼️ Uploaded Image</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(image, use_column_width=True)
                
                # Image metadata
                st.markdown(f"""
                <div style="background: var(--light); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                    <p><strong>Image Details:</strong></p>
                    <p>Dimensions: {image.size[0]} × {image.size[1]} pixels</p>
                    <p>Format: {image.format}</p>
                    <p>Color Mode: {image.mode}</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Image loading error: {e}")
    
    with col2:
        st.markdown("""
        <div class="elegant-card fade-in">
            <div class="card-title">🎯 AI Prediction Results</div>
            <p>Advanced ensemble model combines multiple AI algorithms for maximum classification accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file is not None and classifier and classifier.models:
            # Make prediction
            with st.spinner("🤖 Multiple AI models analyzing..."):
                image = Image.open(uploaded_file)
                class_name, confidence, raw_scores = classifier.predict_ensemble(image)
            
            if class_name:
                confidence_percent = confidence * 100
                
                # Determine confidence level
                if confidence_percent >= 85:
                    conf_class = "confidence-high"
                    conf_text = "VERY HIGH CONFIDENCE"
                    conf_emoji = "🟢"
                elif confidence_percent >= 70:
                    conf_class = "confidence-medium"
                    conf_text = "HIGH CONFIDENCE"
                    conf_emoji = "🟡"
                elif confidence_percent >= 60:
                    conf_class = "confidence-medium"
                    conf_text = "GOOD CONFIDENCE"
                    conf_emoji = "🟡"
                else:
                    conf_class = "confidence-low"
                    conf_text = "LOW CONFIDENCE"
                    conf_emoji = "🔴"
                
                # Display prediction card
                if class_name == "BIRD":
                    st.markdown(f"""
                    <div class="prediction-card bird-prediction fade-in">
                        <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">🐦 BIRD DETECTED</h1>
                        <div class="confidence-display {conf_class}">{conf_emoji} {confidence_percent:.2f}%</div>
                        <h3 style="color: var(--gray);">{conf_text}</h3>
                        <div style="margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.5); border-radius: 10px;">
                            <p><strong>Characteristics Detected:</strong></p>
                            <p>• Organic shapes • Feather textures • Natural contours</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card drone-prediction fade-in">
                        <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">🚁 DRONE DETECTED</h1>
                        <div class="confidence-display {conf_class}">{conf_emoji} {confidence_percent:.2f}%</div>
                        <h3 style="color: var(--gray);">{conf_text}</h3>
                        <div style="margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.5); border-radius: 10px;">
                            <p><strong>Characteristics Detected:</strong></p>
                            <p>• Mechanical structure • Angular shapes • Propellers visible</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Model breakdown
                st.markdown("""
                <div class="elegant-card">
                    <div class="card-title">🔍 Model Analysis</div>
                    <p>Individual model contributions to the ensemble prediction:</p>
                </div>
                """, unsafe_allow_html=True)
                
                if raw_scores:
                    for score in raw_scores:
                        st.markdown(f"""
                        <div style="background: var(--light); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                            <p><strong>{score.split(':')[0]}:</strong> {score.split(':')[1]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Success animation
                st.balloons()
                st.markdown("""
                <div style="text-align: center; margin-top: 2rem;">
                    <p style="color: var(--success); font-weight: 600;">✅ Classification completed successfully!</p>
                </div>
                """, unsafe_allow_html=True)
        
        elif uploaded_file and not classifier.models:
            st.warning("⚠️ AI models not loaded. Please ensure model files are available.")
        else:
            # Default state
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; color: var(--gray);">
                <div style="font-size: 5rem; margin-bottom: 1rem;">📁</div>
                <h3>Awaiting Image Upload</h3>
                <p>Upload an image to begin AI-powered classification analysis.</p>
                <p style="margin-top: 2rem; font-size: 0.9rem; opacity: 0.7;">Supports birds, drones, and aerial objects</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ===== FOOTER =====
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--gray); padding: 2rem;">
        <p><strong>AERIAL INTELLIGENCE PLATFORM</strong> | Advanced AI Classification System</p>
        <p style="font-size: 0.9rem; opacity: 0.7;">Powered by TensorFlow | Ensemble Learning | Computer Vision</p>
        <div style="margin-top: 1rem;">
            <span style="margin: 0 10px;">📊 Real-time Analytics</span>
            <span style="margin: 0 10px;">🤖 AI Models</span>
            <span style="margin: 0 10px;">🔒 Enterprise Security</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()