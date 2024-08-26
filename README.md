# SEMANTIC SEGMENTATION OF DRONE UAV IMAGES TO IMPROVE POST-DISASTER MANAGEMENT

Semantic segmentation of drone UAV images to improve disaster management by detecting and segmenting flood-affected areas. The model is trained on the Floodnet dataset.

## Table of Contents
- [Project Description](#project-description)
- [Live Demonstration](#demonstration)
- [Dataset](#dataset)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Description
This project aims to enhance disaster management by using a deep learning model to perform semantic segmentation on drone images. The model helps in identifying and segmenting flood-affected areas, providing valuable information for response and recovery efforts. It can classify the given image into 10 classes - ('Background':0, 'Building-flooded':1, 'Building-non-flooded':2, 'Road-flooded':3, 'Road-non-flooded':4, 'Water':5, 'Tree':6, 'Vehicle':7, 'Pool':8, 'Grass':9)

## Demonstration
The Streamlit app provides an interface to upload images and view the segmentation results in real-time.
Streamlit Demonstration - [Click here](https://floodnet-implementation.streamlit.app/)


## Dataset
The model is trained on the Floodnet dataset. You can download the dataset from [Floodnet Dataset]([https://floodnet.org/dataset](https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&e=3&dl=0)). Please also refer to the research paper [paper](https://ieeexplore.ieee.org/document/9460988) .


## Results
We used Iou per each class to measure the accuracy of our model. 
<img width="578" alt="image" src="https://github.com/user-attachments/assets/19fbb2d7-fa5b-42d6-ba59-b00aa9d5d581">

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
