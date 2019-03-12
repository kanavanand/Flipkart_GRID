import pandas as pd

xcep=pd.read_csv('regression.csv')

seg = pd.read_csv('segment.csv')



def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


seg.loc[seg.x1.isnull()]=xcep.loc[seg.loc[seg.x1.isnull()].index]


from tqdm import tqdm
from tqdm import tqdm, tqdm_notebook


col=['x1', 'y1', 'x2','y2']

def area(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	
 
	# return the intersection over union value
	return boxAArea>boxBArea



import numpy as np


a=[]
for i in tqdm_notebook(range(len(xcep))):
    sc = bb_intersection_over_union(xcep[col].values[i],seg[col].values[i])
    if sc <0.7:
        if area(xcep[col].values[i],seg[col].values[i]):
            a.append(seg[col].values[i])
        else:
            a.append(xcep[col].values[i])
    else:
        a.append(0.5*xcep[col].values[i]+0.5*seg[col].values[i])

df=pd.DataFrame(a)
df.columns=['x1','y1','x2','y2']


xcep[['x1','x2','y1','y2']]=df[['x1','x2','y1','y2']]

xcep.to_csv('seg_reg_ensemble.csv',index=None)