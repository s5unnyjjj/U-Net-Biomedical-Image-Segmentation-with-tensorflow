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
<img src = "https://user-images.githubusercontent.com/70457520/184366315-34267b21-d9a1-4633-9649-dcd856121b61.png" width="50%" height="50%">


## Validation
```
python train.py
 > Session().validation()
```
 * Validataion result at some epochs
<figure class="half">
    <img src="http://xxx.jpg">
    <img src="http://yyy.jpg">
figure>
