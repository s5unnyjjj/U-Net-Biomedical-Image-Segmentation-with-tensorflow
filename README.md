# U-Net-Biomedical-Image-Segmentation-with-tensorflow </br>
This project is written code based on reading of 'U-Net: Convolutional Networks for Biomedical Image Segmentation' paper.
 > Reference: https://arxiv.org/abs/1505.04597

## Prerequisite </br>
 * Python >= 3.6</br>
 * Tensorflow >= 2.x</br>
 * Opencv</br>
 * Pillow</br>


## Project Structure </br>
 * code: contains all codes
   * train.py: Code to train the u-net model for predicting the label image
   * model.py: Build the architecture of u-net model
   * settting.py: Assign value of various parameters
   * preprocessing.py: Argumentation for changing the size of datasets from input image to patch image
   

## Dataset </br>
Uses biomedical images as dataset, which are saved in 'dataset' folder. It has three folders: train, validataion, test


## Traininig
```
python train.py
 > Session().training()
```
 * Training loss graph
![Loss_graph](https://user-images.githubusercontent.com/70457520/184365069-ddef8580-6a3d-4bfb-94a6-2d7ca62180c0.png){: width="100" height="100"}

## Validation
```
python train.py
 > Session().validation()
```
