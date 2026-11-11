from PIL import Image
import numpy as np, cv2
c=np.asarray(Image.open('assets/gen/E_r2/top_down_gcut.png')).copy()
al=c[...,3].astype(np.float32)/255
h,w=al.shape
b=(al>0.02).astype(np.uint8); b[:,1610:]=0
n,lab,st,cen=cv2.connectedComponentsWithStats(b,8)
big=np.argmax(st[1:,4])+1
keep=np.zeros(n,bool); keep[big]=True
for i in range(1,n):
    if cen[i][0]<1460 or cen[i][1]<820: keep[i]=True
m=keep[lab].astype(np.float32)
# dirt pebbles that touch body? report
print('kept',keep.sum(),'of',n-1)
body=c.copy(); body[...,3]=(al*m*255).astype(np.uint8)
Image.fromarray(body).save('assets/gen/R6B_E/top_down_cut.png')
# sword: above y=885 & x>1640 ; plus blade strip
yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
up=(yy<885)&(xx>1640)
g=np.array([1845,740.]); tip=np.array([1765,1110.])
d=tip-g; L=np.linalg.norm(d); d/=L; nrm=np.array([-d[1],d[0]])
rx=xx-g[0]; ry=yy-g[1]
s=rx*d[0]+ry*d[1]; q=np.abs(rx*nrm[0]+ry*nrm[1])
strip=(s>0)&(s<L)&(q<40)
sm=(up|strip).astype(np.float32)
sa=al*sm
sw=c.copy(); sw[...,3]=(sa*255).astype(np.uint8)
Image.fromarray(sw).save('/tmp/e6/sword_full.png')
bgc=np.array([0,150,255.])
for nm,arr in (('body',body),('sword',sw)):
    a=arr[...,3:].astype(np.float32)/255
    o=arr[...,:3]*a+bgc*(1-a)
    Image.fromarray(o.astype(np.uint8)).resize((1280,720)).save('/tmp/e6/chk_%s.jpg'%nm)
# 发丝去网：头发区（y<620）把细于 ~9px 的飞丝（逆光下读成发光铁丝圈）去掉，保留发束外形，外沿 2px 羽化
im = np.asarray(Image.open('assets/gen/R6B_E/top_down_cut.png')).copy()
a = im[..., 3].astype(np.float32) / 255
sol = (a > 0.5).astype(np.uint8)
op = cv2.morphologyEx(sol, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
env = cv2.GaussianBlur(op.astype(np.float32), (0, 0), 2.0)
yy = np.mgrid[0:a.shape[0], 0:a.shape[1]][0].astype(np.float32)
wgt = np.clip((700 - yy) / 80, 0, 1)
a2 = a * (1 - wgt) + np.minimum(a, env) * wgt
im[..., 3] = (a2 * 255).astype(np.uint8)
Image.fromarray(im).save('assets/gen/R6B_E/top_down_cut.png')
