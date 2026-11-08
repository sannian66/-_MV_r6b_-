import numpy as np,sys
from PIL import Image
from scipy import ndimage
src,out,x0,x1=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
im=np.asarray(Image.open(src).convert('RGB')).astype(np.float32)[:,x0:x1]
mn=im.min(2);mx=im.max(2)
bg=((255-mn)<22)&((mx-mn)<18)
lab,_=ndimage.label(bg);border=set(np.unique(np.r_[lab[0],lab[-1],lab[:,0],lab[:,-1]]))-{0}
bgc=np.isin(lab,list(border))
# small enclosed white islands too (hair gaps), but not large interior (eyes etc are small... restrict to near-border region)
sizes=ndimage.sum(bg,lab,range(lab.max()+1))
small=np.isin(lab,[i for i,s in enumerate(sizes) if 0<s<4000])
near=ndimage.distance_transform_edt(~bgc)<120
key=bgc|(small&near)
zone=ndimage.binary_dilation(key,iterations=10)
lin=np.clip((250-mx)/215,0,1)      # dark-on-white linear alpha
alpha=np.where(key,0,np.where(zone,lin,1.0))
alpha=ndimage.gaussian_filter(alpha,0.6)
A=np.clip(alpha,1e-3,1)[...,None]
rgb=np.clip((im-255*(1-A))/A,0,255)
Image.fromarray(np.dstack([rgb,alpha*255]).astype(np.uint8)).save(out)
