# U-Net-Biomedical-Image-Segmentation-with-tensorflow </br>
This project is written code based on reading of 'U-Net: Convolutional Networks for Biomedical Image Segmentation' paper.
 > Reference: https://arxiv.org/abs/1505.04597
</br>

## Prerequisite </br>
 * Python >= 3.6</br>
 * Tensorflow >= 2.x</br>
 * Opencv</br>
 * Pillow</br>

## Project Structure </br>
 * train.py: Code to train the u-net model for predicting the label image
 * model.py: Build the architecture of u-net model
 * settting.py: Assign value of various parameters
 * preprocessing.py: Argumentation for changing the size of datasets from input image to patch image

## Dataset </br>
Uses biomedical images as dataset, which are saved in 'dataset' folder. It has three folders: train, validataion, test


## Traininig </br>
```
python train.py
 > Session().training()
```
 * Training loss graph
<img src = "https://user-images.githubusercontent.com/70457520/184366315-34267b21-d9a1-4633-9649-dcd856121b61.png" width="50%" height="50%">
</br>

## Validation
```
python train.py
 > Session().validation()
```
 * Validataion results at some iterations </br>

|Iter 100|Iter 200|Iter 300|
|:---:|:---:|:---:| 
|<img src="https://user-images.githubusercontent.com/70457520/184367132-8207dadc-84c0-4627-8c21-907bd364ea2d.png" width="380" height="350">|<img src="https://user-images.githubusercontent.com/70457520/184367223-bb6a6945-6f1d-4cb5-b418-ddc907355c36.png" width="380" height="350">|<img src="https://user-images.githubusercontent.com/70457520/184367259-8bb46583-3bb5-4de4-8af9-9927d0af2065.png" width="380" height="350">| 

|Iter 400|Iter 500|Iter 2000|
|:---:|:---:|:---:| 
|<img src="https://user-images.githubusercontent.com/70457520/184367329-0f89e345-0aa7-4847-996f-4714e3e491b8.png" width="380" height="350">|<img src="https://user-images.githubusercontent.com/70457520/184367422-026fa515-8562-4cce-a19a-7f2c31e12436.png" width="380" height="350">|<img src="https://user-images.githubusercontent.com/70457520/184367452-fe7929d7-2706-4fca-b8c7-156a00a4eb7b.png" width="380" height="350">| 
