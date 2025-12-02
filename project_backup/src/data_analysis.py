import os
import sys

# Try to import with error handling
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from PIL import Image
    import numpy as np
    print("✅ All packages imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages using: pip install matplotlib seaborn pandas pillow numpy")
    sys.exit(1)

class DataAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.results = {}
        
    def analyze_dataset(self):
        print("🔍 Analyzing Dataset Structure...")
        
        # Define paths
        paths = {
            'train_bird': os.path.join(self.base_path, 'train', 'bird'),
            'train_drone': os.path.join(self.base_path, 'train', 'drone'),
            'valid_bird': os.path.join(self.base_path, 'valid', 'bird'),
            'valid_drone': os.path.join(self.base_path, 'valid', 'drone'),
            'test_bird': os.path.join(self.base_path, 'test', 'bird'),
            'test_drone': os.path.join(self.base_path, 'test', 'drone')
        }
        
        # Count images in each directory
        for name, path in paths.items():
            if os.path.exists(path):
                num_images = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                self.results[name] = num_images
                print(f"📁 {name}: {num_images} images")
            else:
                print(f"❌ Path not found: {path}")
                
        return self.results
    
    def plot_class_distribution(self):
        print("\n📊 Plotting Class Distribution...")
        
        try:
            # Prepare data for plotting
            train_bird = self.results.get('train_bird', 0)
            train_drone = self.results.get('train_drone', 0)
            valid_bird = self.results.get('valid_bird', 0)
            valid_drone = self.results.get('valid_drone', 0)
            test_bird = self.results.get('test_bird', 0)
            test_drone = self.results.get('test_drone', 0)
            
            # Create subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Train distribution
            axes[0].pie([train_bird, train_drone], labels=['Bird', 'Drone'], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
            axes[0].set_title('Train Set Distribution')
            
            # Validation distribution
            axes[1].pie([valid_bird, valid_drone], labels=['Bird', 'Drone'], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
            axes[1].set_title('Validation Set Distribution')
            
            # Test distribution
            axes[2].pie([test_bird, test_drone], labels=['Bird', 'Drone'], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
            axes[2].set_title('Test Set Distribution')
            
            plt.tight_layout()
            plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
            print("✅ Class distribution chart saved as 'class_distribution.png'")
            plt.close()
            
        except Exception as e:
            print(f"❌ Error in plotting: {e}")
        
    def display_sample_images(self):
        print("\n🖼️ Displaying Sample Images...")
        
        try:
            sample_paths = [
                os.path.join(self.base_path, 'train', 'bird'),
                os.path.join(self.base_path, 'train', 'drone')
            ]
            
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle('Sample Images from Dataset', fontsize=16)
            
            images_found = False
            
            for i, path in enumerate(sample_paths):
                if os.path.exists(path):
                    images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][:5]
                    if images:
                        images_found = True
                        for j, img_name in enumerate(images):
                            img_path = os.path.join(path, img_name)
                            try:
                                img = Image.open(img_path)
                                axes[i, j].imshow(img)
                                axes[i, j].set_title(f"{'Bird' if i==0 else 'Drone'}\n{img_name[:15]}...")
                                axes[i, j].axis('off')
                            except Exception as e:
                                print(f"Error loading image {img_path}: {e}")
                                axes[i, j].text(0.5, 0.5, 'Image\nLoad Error', 
                                              ha='center', va='center', transform=axes[i, j].transAxes)
                                axes[i, j].axis('off')
                else:
                    print(f"Path not found: {path}")
            
            if images_found:
                plt.tight_layout()
                plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
                print("✅ Sample images saved as 'sample_images.png'")
            else:
                print("❌ No sample images found to display")
                
            plt.close()
            
        except Exception as e:
            print(f"❌ Error displaying sample images: {e}")
    
    def create_summary_report(self):
        print("\n📋 Creating Summary Report...")
        
        total_images = sum(self.results.values())
        print(f"\n📊 DATASET SUMMARY:")
        print(f"Total Images: {total_images}")
        print(f"Training Images: {self.results.get('train_bird', 0) + self.results.get('train_drone', 0)}")
        print(f"Validation Images: {self.results.get('valid_bird', 0) + self.results.get('valid_drone', 0)}")
        print(f"Test Images: {self.results.get('test_bird', 0) + self.results.get('test_drone', 0)}")
        
        # Save summary to text file
        with open('dataset_summary.txt', 'w') as f:
            f.write("AERIAL OBJECT CLASSIFICATION DATASET SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Images: {total_images}\n")
            f.write(f"Training Set: {self.results.get('train_bird', 0) + self.results.get('train_drone', 0)} images\n")
            f.write(f"Validation Set: {self.results.get('valid_bird', 0) + self.results.get('valid_drone', 0)} images\n")
            f.write(f"Test Set: {self.results.get('test_bird', 0) + self.results.get('test_drone', 0)} images\n")
            f.write("\nDetailed Breakdown:\n")
            for key, value in self.results.items():
                f.write(f"{key}: {value} images\n")
        
        print("✅ Summary report saved as 'dataset_summary.txt'")

def main():
    # Set your data path here
    base_path = r"C:\JN\Aerial_Object_Classification\data\classification_dataset"
    
    print("🚀 Starting Data Analysis...")
    print(f"Data path: {base_path}")
    
    if not os.path.exists(base_path):
        print(f"❌ Data path not found: {base_path}")
        print("Please check the path and try again.")
        input("Press Enter to exit...")
        return
    
    analyzer = DataAnalyzer(base_path)
    
    # Perform analysis
    print("\n" + "="*50)
    analyzer.analyze_dataset()
    print("\n" + "="*50)
    analyzer.plot_class_distribution()
    print("\n" + "="*50)
    analyzer.display_sample_images()
    print("\n" + "="*50)
    analyzer.create_summary_report()
    
    print("\n🎉 Data analysis completed successfully!")
    print("📁 Generated files:")
    print("   - class_distribution.png")
    print("   - sample_images.png") 
    print("   - dataset_summary.txt")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()