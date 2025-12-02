import sys

def test_tensorflow():
    print("Testing TensorFlow installation...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check GPU availability
        gpu_available = tf.config.list_physical_devices('GPU')
        if gpu_available:
            print("✅ GPU is available!")
        else:
            print("ℹ️  Using CPU (GPU not available)")
            
        # Test basic operations
        print("Testing basic operations...")
        a = tf.constant([[1, 2], [3, 4]])
        b = tf.constant([[5, 6], [7, 8]])
        c = tf.matmul(a, b)
        print(f"✅ Matrix multiplication test: {c.numpy()}")
        
        print("\n🎉 TensorFlow is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow error: {e}")
        return False

if __name__ == "__main__":
    success = test_tensorflow()
    if not success:
        print("\nTroubleshooting tips:")
        print("1. Run: pip install tensorflow-cpu")
        print("2. Run: pip install tensorflow==2.13.0")
        print("3. Restart your command prompt")
    
    input("\nPress Enter to exit...")