from PIL import Image
import numpy as np, cv2
c=np.asarray(Image.open('assets/gen/E_r2/top_down_gcut.png')).copy()
al=c[...,3].astype(np.float32)/255
h,w=al.shape
yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
up=((yy<820)&(xx>1640)).astype(np.float32)
q=np.zeros((h,w),np.float32)
cv2.fillPoly(q,[np.array([[1792,770],[1865,770],[1771,1115],[1741,1115]],np.int32)],1.0)
q=cv2.GaussianBlur(q,(0,0),1.0)
fade=np.clip((1115-yy)/30,0,1)
rb=c[...,0].astype(np.float32)-c[...,2].astype(np.float32)
steel=np.clip(1-(rb-14)/14,0,1)
steel=cv2.erode(steel,np.ones((5,5),np.uint8))
sm=np.maximum(up,q*fade)
lum=c[...,:3].astype(np.float32).mean(-1,keepdims=True)
bad=((1-steel)*(yy>840)*(q>0))[...,None]
gray=np.clip(lum*np.array([1.0,0.99,0.97])*0.85,0,255)
c=c.copy(); c[...,:3]=(c[...,:3]*(1-bad)+gray*bad).astype(np.uint8)
sw=c.copy(); sw[...,3]=(al*sm*255).astype(np.uint8)
sw=sw[400:1140,1640:2420]
Image.fromarray(sw).save('assets/gen/R6B_E/sword_td_cut.png')
a=sw[...,3:].astype(np.float32)/255
o=sw[...,:3]*a+np.array([20,16,12.])*(1-a)
Image.fromarray(o.astype(np.uint8)).save('/tmp/e6/chk_sword.png')
