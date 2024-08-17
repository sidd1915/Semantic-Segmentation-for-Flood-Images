import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Define your color map for classes
color_map = {
    0: [255, 255, 255],    # Class 0: White
    1: [255, 0, 0],        # Class 1: Red
    2: [0, 255, 0],        # Class 2: Green
    3: [0, 0, 255],        # Class 3: Blue
    4: [255, 255, 0],      # Class 4: Yellow
    5: [0, 255, 255],      # Class 5: Cyan
    6: [0, 128, 0],        # Class 6: Dark Green
    7: [128, 0, 128],      # Class 7: Purple
    8: [0, 128, 128],      # Class 8: Teal
    9: [128, 128, 0]       # Class 9: Olive
}

class_names = {
    0: 'Background',
    1: 'Building-flooded',
    2: 'Building-non-flooded',
    3: 'Road-flooded',
    4: 'Road-non-flooded',
    5: 'Water',
    6: 'Tree',
    7: 'Vehicle',
    8: 'Pool',
    9: 'Grass'
}

model = tf.keras.models.load_model('floodnet_model_25.h5', compile=False)

def display_color_map_legend():
    
    legend_image = np.zeros((len(color_map) * 30, 150, 3), dtype=np.uint8)

    
    for idx, color in color_map.items():
        # Color stripe
        legend_image[idx * 30:(idx + 1) * 30, :30] = color

        # Class name and description text
        class_name = class_names[idx]
        class_description = f'{class_name}: {idx}'
        cv2.putText(legend_image, class_description, (40, idx * 30 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    
    st.sidebar.image(legend_image, caption='Class Color Map Legend', use_column_width=True)



# Function to process image into patches and stitch them back
def segment_image(image):
    # Convert image to numpy array
    img_array = np.array(image)

    # Get image dimensions
    img_height, img_width, _ = img_array.shape

    # Initialize results array with zeros for 10 classes
    results = np.zeros((img_height, img_width, 10))

    patch_size = 512
    stride = 256  # Overlap between patches
    for y in range(0, img_height - patch_size + 1, stride):
        for x in range(0, img_width - patch_size + 1, stride):
            
            patch = img_array[y:y+patch_size, x:x+patch_size, :]

            
            patch_result = model.predict(np.expand_dims(patch, axis=0))

            
            results[y:y+patch_size, x:x+patch_size, :] += patch_result[0]

    
    segmented_image = np.argmax(results, axis=-1)

    
    rgb_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        rgb_image[segmented_image == class_idx] = color

    return rgb_image

# Streamlit UI
def main():
    st.title("Semantic Segmentation Web App")
    st.sidebar.title("Options")
    
    display_color_map_legend()

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
       
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        
        if st.sidebar.button('Segment'):
            
            st.write("Segmenting...")
            img_array = np.array(image)      
            segmented_image = segment_image(img_array)
            st.image(segmented_image, caption='Segmented Image.', use_column_width=True)

if __name__ == '__main__':
    main()
