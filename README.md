# Croppy [Flipkart GRID](https://dare2compete.com/o/Flipkart-GRiD-Teach-The-Machines-2019-74928)
The code is used to predict the bounding box for an object in image. This is keras implementation of single object localization. It's based on UNET segmentation and regression both based on XceptionNet. 
![alt text](Screen Shot 2019-03-13 at 12.18.59 PM.png)
The croppy supports two modes currently 
1) Regression Mode :"reg"
2) Segmentation Mode : "seg"


#### 1:  Regression 
The [XceptionNet](https://keras.io/applications/#xception) is build using keras api with output layer giving 4 coordinates of bounding box [x1,y1,x2,y2].
The regression model got an IOU score of 93.7 on GRID dataset. You can use pre_trained_reg.py or train it again on your dataset using train.py with "reg" mode in config.py

#### 2:  Segmentation  
The [UNet](https://arxiv.org/abs/1505.04597) is also build using keras api with output layer giving an mask of (224,224,1) . Further using cv2.regionprops we can get the desired bounding box coordinates.
The Segmentation model alone got an IOU score of 94.12 on GRID dataset. You can use pre_trained_seg.py or train it again on your dataset using train.py with "seg" mode in config.py .

## Installation

1. Download the [zip](https://drive.google.com/open?id=1ef-NATi1PV9XdQhuLt1YTzlDIZzYHyu7) file in your current working directory.
2. you need to run following command in shell a directory 'pretrained_weights/' download the weights in this directory.

```python
import os
if not os.path.isdir("pretrained_weights/"):
    print('making direct pretrained_weights/')
    os.mkdir('pretrained_weights/')
! wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Lx9uZSwbzsc3anh9lfb4RPiBEBoSDMQ8' -O pretrained_weights/regression.h5
! wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WNe1s1RzBqiV-vvcJMo9swaAFDBO1Rh-' -O pretrained_weights/segmentation.h5

```

## Usage
You need to change the config.py file according to your use.
```python
img_size=(224,224,3) #input size 
mode='reg'   #The mode in which you want to run "seg": segmantation and "reg": Regression
model_direc="pretrained_weights/regression.h5"
bs=16  #batch size 
epochs=10   #epochs 
training_df='training_set.csv'   #Location of training dataframe
testing_df = 'test_updated.csv'  #Location of testing dataframe 
test_direc = "testing_mini/"    #Folder containing test images
image_direc="images/"           #Folder containing training images
```
After you have modified the config file you can run python code using shell. For -eg.
1. Training model 
```bash
python train.py
```
2. Predicting the bounding boxes for test_direc using pretrained wieghts in reg mode.
```bash
python pre_trained_reg.py
```
3. Predicting the bounding boxes for test_direc using pretrained wieghts in seg mode.
```bash
python pre_trained_seg.py
```
###### Note : Be carefull, give model_direc = regression weights when in reg mode and same goes for seg mode. Interchanging models will throw an error.
