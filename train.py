from model_unet import*
from model_light import*
from config import*
import pandas as pd
from keras.utils import Sequence


if mode=='seg':
    model=build_model(img_size)
else:
    model = build_regess_model()

im=img_size[0]

img_shape  = img_size
train = pd.read_csv(training_df)
test = pd.read_csv(testing_df)


class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            callbacks.ModelCheckpoint("keras_temp.model",monitor='val_my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1),
            swa,
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)

      
      
class SWA(keras.callbacks.Callback):
    
    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch 
    
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
            
        elif epoch > self.swa_epoch:    
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] * 
                    (epoch - self.swa_epoch) + self.model.get_weights()[i])/((epoch - self.swa_epoch)  + 1)  

        else:
            pass
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')
      


from parameters import*
sgd = get_optimizer()
snapshot = SnapshotCallbackBuilder(nb_epochs=epochs,nb_snapshots=1,init_lr=1e-3)
batch_size = bs
print("Training a model with batch size : "+str(bs)+"and epochs: "+str(epochs))
swa = SWA('keras_swa_temp'+mode+'.model',np.round(epochs/2)+1)

smooth=1
def my_iou_metric(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def IOU_calc_loss(y_true, y_pred):
    return -my_iou_metric(y_true, y_pred)

if mode=='seg':
    model.compile(loss=IOU_calc_loss, optimizer=sgd, metrics=[my_iou_metric])
else:
    model.compile(loss=combine_loss,optimizer=Adam(),metrics=[my_metric,smooth_l1_loss])
        
print("loading model weights from ",model_direc)
model.load_weights(model_direc)



# define the size of each image you want to send to model for training.
train['x1']=train['x1']*(224/640)
train['x2']=train['x2']*(224/640)
train['y1']=train['y1']*(224/480)
train['y2']=train['y2']*(224/480)
data=train[['image_name', 'x1', 'y1', 'x2','y2']].values

train, val = train_test_split(data, test_size=500, random_state=1)
len(train),len(val)
val_a = np.zeros((len(val),)+img_shape,dtype=K.floatx()) # Preprocess validation images 
if mode=='seg':
        val_b = np.zeros((len(val),)+(224,224,1),dtype=K.floatx()) 
else:
    val_b = np.zeros((len(val),4),dtype=K.floatx())
# Preprocess bounding boxes
for i,j in enumerate(tqdm(val)):
    img   = read_for_validation(j[0])
    val_a[i,:,:,:] = img
    if mode=='seg':
        val_b[i,:,:,:] = get_mask_seg(img,j[1:])
if mode =='reg':
    val_b=val[:,1:] 
    
    
class TrainingData(Sequence):
    def __init__(self, batch_size=32):
        super(TrainingData, self).__init__()
        self.batch_size = batch_size
    def __getitem__(self, index):
        start = self.batch_size*index;
        end   = min(len(train), start + self.batch_size)
        size  = end - start
        a     = np.zeros((size,) + img_shape, dtype=K.floatx())
        b     = np.zeros((size,4), dtype=K.floatx())
        b2     = np.zeros((size,) + (224,224,1), dtype=K.floatx())
        for i,j in enumerate(train[start:end]):
            img  = read_for_training(j[0])
            a[i,:,:,:]  = img
            b2[i,:,:,:]  = get_mask_seg(img,list(j[1:]))
        b=train[start:end,1:] 
        if mode=='seg':
            return a,b2
        else:
            return a,b
    def __len__(self):
        return (len(train) + self.batch_size - 1)//self.batch_size    

history = model.fit_generator(
        TrainingData(bs),epochs=epochs, max_queue_size=12, workers=4, verbose=1,
        validation_data=(val_a, val_b),
                    callbacks=snapshot.get_callbacks(),shuffle=True)
