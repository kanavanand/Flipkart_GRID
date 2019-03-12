from utils import*
from keras.applications import*
from keras.applications import resnet50
img_shape=(224,224,3)
def build_regess_model(img_size=img_shape):
    inp = Input(shape=img_size)
    x = inp
    #### using exception models with weights =NONE(random)
    base_model=Xception(weights=None,include_top=False,input_tensor=inp,input_shape=img_size) 

    for layer in base_model.layers:
            layer.trainable = True

    x=base_model.output
    y=GlobalAveragePooling2D()(x)
    y=Dense(1000,kernel_initializer='he_normal',)(y)
    outputs=Dense(4,kernel_initializer='he_normal',)(y)
    model=Model(inputs=base_model.input,outputs=outputs)
    return model


