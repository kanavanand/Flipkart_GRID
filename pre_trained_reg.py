from model_light import*
from tqdm import tqdm


from config import*
model = build_regess_model()


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
    a  = np.expand_dims(img, axis=0)
    bbx=change_t2(change_t(model.predict(a).squeeze()))
    pred.append(bbx)


df = pd.DataFrame(pred,index=st)
df.reset_index(inplace=True)
df.columns=['image_name','x1','x2','y1','y2']
print("check your folder a new file containing co-ordinats is created in working dir")
df.to_csv('regression.csv',index=None)