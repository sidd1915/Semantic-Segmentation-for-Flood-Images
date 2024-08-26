import streamlit as st
import os
import random
from PIL import Image

# Function to load images with extension handling
def load_image(file_path):
    if os.path.exists(file_path + '.png'):
        return Image.open(file_path + '.png')
    elif os.path.exists(file_path + '.jpg'):
        return Image.open(file_path + '.jpg')
    else:
        raise FileNotFoundError(f"No file found for {file_path} with .jpg or .png extension.")

def load_images(folder, index):
    test_image = load_image(os.path.join(folder, f'test{index}'))
    result_image = load_image(os.path.join(folder, f'result{index}'))
    return test_image, result_image

# Main Streamlit App
def main():
    st.title("Post-Disaster Assesment using Deep Learning")

    # Sidebar content
    st.sidebar.title("Floodnet")
    sidebar_image = st.sidebar.image('Labels.png', caption="Label Colouring", use_column_width=True)
    st.sidebar.write("This is a demonstration for our implementation for semantic segmentation of floodnet dataset. We implemented a Unet architecture with Vgg16 as pretrained model.")

    if st.button("Generate Random Set"):
        try:
            # Randomly select an index between 1 and 7
            random_index = random.randint(1, 7)
            
            # Load the corresponding images
            test_image, result_image = load_images('Results', random_index)

            # Display the images side by side
            col1, col2 = st.columns(2)

            with col1:
                st.image(test_image, caption="Input Image")

            with col2:
                st.image(result_image, caption="Prediction")

        except FileNotFoundError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")

if __name__ == "__main__":
    main()
