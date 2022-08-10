# U-Net-Biomedical-Image-Segmentation-with-tensorflow

## Prerequisite </br>
 * Python >= 3.6</br>
 * Tensorflow >= 2.x</br>
 * Opencv</br>
 * Pillow</br>

## Project Structure </br>
 * code: contains all codes
   * train.py
   * model.py
   * settting.py: 
   * preprocessing.py

## Dataset </br>
Uses biomedical images as dataset, which are saved in 'dataset' folder. It has three folders: train, validataion, test


## Traininig
```
python train.py
 > Session().training()
```

## Validation
```
python train.py
 > Session().validation()
```
