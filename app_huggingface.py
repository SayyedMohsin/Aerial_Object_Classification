import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Aerial Intelligence",
    page_icon="🛸",
    layout="wide"
)

# Professional CSS
st.markdown("""
<style>
.header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 2rem;
    border-radius: 0 0 20px 20px;
    text-align: center;
    margin-bottom: 2rem;
}
.header h1 {
    font-size: 2.5rem;
    margin: 0;
    background: linear-gradient(to right, #fff 0%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.prediction-card {
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin: 1rem 0;
    border: 3px solid;
}
.bird-card {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    border-color: #10b981;
}
.drone-card {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-color: #f59e0b;
}
.confidence {
    font-size: 2.5rem;
    font-weight: bold;
    margin: 1rem 0;
}
.high { color: #10b981; }
.medium { color: #f59e0b; }
.low { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# Load model with multiple format support
@st.cache_resource
def load_model():
    """Load model in Hugging Face compatible way"""
    try:
        # Try different formats and paths
        model_paths = [
            'hf_model.keras',
            'hf_model.h5',
            'models/hf_model.keras',
            'models/hf_model.h5',
            'final_model.h5',
            'models/final_model.h5'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                st.success(f"Found model: {path}")
                model = tf.keras.models.load_model(path)
                st.success("Model loaded successfully!")
                return model
        
        # If no model found, create a simple one
        st.warning("No model file found. Creating simple model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(150, 150, 3)),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
        
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Main app
def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>🛸 Aerial Object Classification</h1>
        <p>AI-powered Bird vs Drone Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="card">
            <h3>System Info</h3>
            <p>TensorFlow: {}</p>
            <p>Status: {}</p>
        </div>
        """.format(tf.__version__, "✅ Active" if model else "❌ Inactive"), unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>📤 Upload Image</h3>
            <p>Upload aerial image for classification</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose image (JPG, JPEG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            st.info(f"Size: {image.size} | Format: {image.format}")
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🎯 Prediction</h3>
            <p>AI classification results</p>
        </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file and model:
            with st.spinner("Analyzing..."):
                # Preprocess
                img = image.resize((150, 150))
                img_array = np.array(img) / 255.0
                
                if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
                    img_array = img_array[:, :, :3]
                
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                prediction = model.predict(img_array, verbose=0)
                raw_score = float(prediction[0][0])
                
                # Determine class
                if raw_score < 0.5:
                    class_name = "BIRD"
                    confidence = 1 - raw_score
                else:
                    class_name = "DRONE"
                    confidence = raw_score
                
                confidence_percent = confidence * 100
                
                # Confidence level
                if confidence_percent >= 85:
                    conf_class = "high"
                    conf_text = "Very High Confidence"
                elif confidence_percent >= 70:
                    conf_class = "medium"
                    conf_text = "High Confidence"
                else:
                    conf_class = "low"
                    conf_text = "Moderate Confidence"
                
                # Display result
                if class_name == "BIRD":
                    st.markdown(f"""
                    <div class="prediction-card bird-card">
                        <h1>🐦 BIRD DETECTED</h1>
                        <div class="confidence {conf_class}">{confidence_percent:.1f}%</div>
                        <p>{conf_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card drone-card">
                        <h1>🚁 DRONE DETECTED</h1>
                        <div class="confidence {conf_class}">{confidence_percent:.1f}%</div>
                        <p>{conf_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success(f"Raw score: {raw_score:.4f}")
                st.balloons()
        
        elif uploaded_file:
            st.error("Model not loaded. Cannot make prediction.")
        else:
            st.info("Upload an image to see predictions")

if __name__ == "__main__":
    main()