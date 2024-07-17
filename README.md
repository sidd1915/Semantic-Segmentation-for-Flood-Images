#SEMANTIC SEGMENTATION OF DRONE CAPTURED IMAGES TO IMPROVE DISASTER MANAGEMENT

# Semantic Segmentation for Flood Images

Semantic segmentation of drone UAV images to improve disaster management by detecting and segmenting flood-affected areas. The model is trained on the Floodnet dataset.

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Description
This project aims to enhance disaster management by using a deep learning model to perform semantic segmentation on drone images. The model helps in identifying and segmenting flood-affected areas, providing valuable information for response and recovery efforts. It can classify the given image into 10 classes - ('Background':0, 'Building-flooded':1, 'Building-non-flooded':2, 'Road-flooded':3, 'Road-non-flooded':4, 'Water':5, 'Tree':6, 'Vehicle':7, 'Pool':8, 'Grass':9)

## Installation
To get started with this project, follow these steps:

1. **Clone the repository**
    ```bash
    git clone https://github.com/sidd1915/Semantic-Segmentation-for-Flood-Images.git
    ```

2. **Install the required dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Navigate to the project directory**
    ```bash
    cd semantic-segmentation-flood-images
    ```

2. **Run the Streamlit app**
    ```bash
    streamlit run app.py
    ```

3. **Access the app**
    - Open your web browser and go to `http://localhost:8501`


## Dataset
The model is trained on the Floodnet dataset. You can download the dataset from [Floodnet Dataset]([https://floodnet.org/dataset](https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&e=3&dl=0)). Please also refer to the research paper [paper](https://ieeexplore.ieee.org/document/9460988) 


## Results
The model achieves an accuracy of 80% on the test set. Below are some example results from the model:
![image](https://github.com/user-attachments/assets/4be77119-896b-4b44-8bd9-32cab05f81a9)

The Streamlit app provides an interface to upload images and view the segmentation results in real-time.

## Contributing
We welcome contributions to enhance this project. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Create a new Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
We would like to thank the developers of the Floodnet dataset and the contributors to open-source deep learning libraries that made this project possible.

Citations for papers referred - 
@ARTICLE{9460988,
 author={Rahnemoonfar, Maryam and Chowdhury, Tashnim and Sarkar, Argho and Varshney, Debvrat and Yari, Masoud and Murphy, Robin Roberson},
 journal={IEEE Access}, 
 title={FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding}, 
 year={2021},
 volume={9},
 number={},
 pages={89644-89654},
 doi={10.1109/ACCESS.2021.3090981}
 }

@article{rahnemoonfar2020floodnet,
 title={FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding},
 author={Rahnemoonfar, Maryam and Chowdhury, Tashnim and Sarkar, Argho and Varshney, Debvrat and Yari, Masoud and Murphy, Robin},
 journal={arXiv preprint arXiv:2012.02951},
 year={2020}
}
