from model_unet import*
from config import*
from tqdm import tqdm

model=build_model(img_size)

img_shape  = img_size


print("loading model weights from ",model_direc)
model.load_weights(model_direc)
sz=img_size[0]
def change_t(boxe):
    a=[]
    a.append(boxe[0]*sz)
    a.append(boxe[1]*sz) 
    a.append(boxe[2]*sz)  
    a.append(boxe[3]*sz) 
    return a

def change_t2(boxe):
  a=[]
  a.append(boxe[1]*640/224)
  a.append(boxe[3]*640/224)
  a.append(boxe[0]*480/224)
  a.append(boxe[2]*480/224) 
  return a



st=os.listdir(test_direc)
pred=[]
print("making predictions")
for i in tqdm(st):
    img=read_for_validation(test_direc+i)
    a = np.expand_dims(img, axis=0)
 
    mask = np.array(np.greater(model.predict(a), 0.7), dtype=np.uint8).squeeze()

    try:
      rgnprops = measure.regionprops(mask)[0]

      x1, y1, x2, y2 = rgnprops.bbox

      pred.append([x1, y1, x2, y2])
    except:
      pred.append([np.NaN,np.NaN,np.NaN,np.NaN])    

bx=[]
for i in tqdm_notebook(pred):
  bx.append(change_t2(i))

df = pd.DataFrame(bx,index=st)
df.reset_index(inplace=True)
df.columns=['image_name','x1','x2','y1','y2']
print("check your folder a new file containing co-ordinats is created in working dir")
df.to_csv('segment.csv',index=None)