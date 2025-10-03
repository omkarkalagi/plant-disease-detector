"""
Demo Data Generator for Plant Disease Detection
Creates synthetic training data when real dataset is not available
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from datetime import datetime

def create_demo_dataset(output_dir="demo_dataset", num_samples_per_class=50):
    """Create a demo dataset with synthetic plant disease images"""
    
    # Plant disease classes based on research
    disease_classes = [
        "Healthy",
        "Bacterial_Spot", 
        "Early_Blight",
        "Late_Blight",
        "Leaf_Mold",
        "Septoria_Leaf_Spot",
        "Spider_Mites",
        "Target_Spot",
        "Mosaic_Virus",
        "Yellow_Leaf_Curl"
    ]
    
    print(f"Creating demo dataset with {len(disease_classes)} classes...")
    
    # Create directories
    for class_name in disease_classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # Generate synthetic images
    for class_idx, class_name in enumerate(disease_classes):
        print(f"Generating {num_samples_per_class} samples for {class_name}...")
        
        for sample_idx in range(num_samples_per_class):
            # Create base image (green leaf-like shape)
            img = Image.new('RGB', (224, 224), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw leaf shape
            leaf_color = (34, 139, 34)  # Forest green
            if class_name != "Healthy":
                # Add disease patterns
                leaf_color = add_disease_pattern(draw, class_name, class_idx)
            
            # Draw main leaf
            draw.ellipse([50, 50, 174, 174], fill=leaf_color, outline='darkgreen', width=2)
            
            # Add leaf veins
            draw.line([112, 50, 112, 174], fill='darkgreen', width=1)
            draw.line([50, 112, 174, 112], fill='darkgreen', width=1)
            
            # Add class label
            try:
                # Try to use a default font
                font = ImageFont.load_default()
            except:
                font = None
            
            text = f"{class_name.replace('_', ' ')}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position text at bottom
            x = (224 - text_width) // 2
            y = 200 - text_height
            draw.text((x, y), text, fill='black', font=font)
            
            # Save image
            filename = f"{class_name}_{sample_idx:03d}.jpg"
            filepath = os.path.join(output_dir, class_name, filename)
            img.save(filepath, 'JPEG', quality=85)
    
    print(f"Demo dataset created successfully in {output_dir}")
    print(f"Total samples: {len(disease_classes) * num_samples_per_class}")
    
    return output_dir

def add_disease_pattern(draw, disease_class, class_idx):
    """Add disease-specific patterns to the leaf"""
    
    # Base healthy color
    base_color = (34, 139, 34)
    
    if disease_class == "Bacterial_Spot":
        # Add small dark spots
        for _ in range(random.randint(5, 15)):
            x = random.randint(60, 164)
            y = random.randint(60, 164)
            size = random.randint(3, 8)
            draw.ellipse([x-size, y-size, x+size, y+size], fill='darkred')
        return (139, 69, 19)  # Brownish
    
    elif disease_class == "Early_Blight":
        # Add concentric rings
        for i in range(3):
            radius = 20 + i * 15
            draw.ellipse([112-radius, 112-radius, 112+radius, 112+radius], 
                        outline='darkred', width=2)
        return (160, 82, 45)  # Saddle brown
    
    elif disease_class == "Late_Blight":
        # Add irregular dark patches
        for _ in range(random.randint(3, 8)):
            x = random.randint(50, 174)
            y = random.randint(50, 174)
            w = random.randint(10, 30)
            h = random.randint(10, 30)
            draw.ellipse([x-w, y-h, x+w, y+h], fill='darkred')
        return (139, 0, 0)  # Dark red
    
    elif disease_class == "Leaf_Mold":
        # Add fuzzy patches
        for _ in range(random.randint(4, 10)):
            x = random.randint(60, 164)
            y = random.randint(60, 164)
            size = random.randint(8, 20)
            draw.ellipse([x-size, y-size, x+size, y+size], fill='gray')
        return (105, 105, 105)  # Dim gray
    
    elif disease_class == "Septoria_Leaf_Spot":
        # Add small circular spots
        for _ in range(random.randint(8, 20)):
            x = random.randint(60, 164)
            y = random.randint(60, 164)
            size = random.randint(2, 6)
            draw.ellipse([x-size, y-size, x+size, y+size], fill='black')
        return (107, 142, 35)  # Olive drab
    
    elif disease_class == "Spider_Mites":
        # Add web-like patterns
        for _ in range(random.randint(3, 6)):
            x1 = random.randint(50, 174)
            y1 = random.randint(50, 174)
            x2 = random.randint(50, 174)
            y2 = random.randint(50, 174)
            draw.line([x1, y1, x2, y2], fill='white', width=1)
        return (85, 107, 47)  # Dark olive green
    
    elif disease_class == "Target_Spot":
        # Add target-like patterns
        for i in range(2):
            radius = 15 + i * 10
            draw.ellipse([112-radius, 112-radius, 112+radius, 112+radius], 
                        outline='darkred', width=3)
        return (160, 82, 45)  # Saddle brown
    
    elif disease_class == "Mosaic_Virus":
        # Add mosaic pattern
        for i in range(0, 224, 20):
            for j in range(0, 224, 20):
                if (i + j) % 40 == 0:
                    draw.rectangle([i, j, i+20, j+20], fill='yellow')
        return (154, 205, 50)  # Yellow green
    
    elif disease_class == "Yellow_Leaf_Curl":
        # Add yellow patches
        for _ in range(random.randint(3, 7)):
            x = random.randint(60, 164)
            y = random.randint(60, 164)
            size = random.randint(15, 35)
            draw.ellipse([x-size, y-size, x+size, y+size], fill='yellow')
        return (255, 255, 0)  # Yellow
    
    return base_color

if __name__ == "__main__":
    # Create demo dataset
    demo_dir = create_demo_dataset("demo_dataset", num_samples_per_class=30)
    print(f"Demo dataset ready at: {demo_dir}")
