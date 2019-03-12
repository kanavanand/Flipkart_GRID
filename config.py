img_size=(224,224,3)

mode='reg'
if mode=='reg':
    model_direc="pretrained_weights/regression.h5"
if mode=='seg':
    model_direc="pretrained_weights/segmentation.model"
bs=16
epochs=10

training_df='training_set.csv'
testing_df = 'test_updated.csv'

test_direc = "testing_mini/"
image_direc="../../images/"
