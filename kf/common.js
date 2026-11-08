// seeded rng + helpers (deterministic)
function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
function hash(x,y){let h=Math.sin(x*127.1+y*311.7)*43758.5453;return h-Math.floor(h);}
function vnoise(x,y){const ix=Math.floor(x),iy=Math.floor(y),fx=x-ix,fy=y-iy;const u=fx*fx*(3-2*fx),v=fy*fy*(3-2*fy);
 const a=hash(ix,iy),b=hash(ix+1,iy),c=hash(ix,iy+1),d=hash(ix+1,iy+1);return a+(b-a)*u+(c-a)*v+(a-b-c+d)*u*v;}
function fbm(x,y,o=5){let s=0,a=.5;for(let i=0;i<o;i++){s+=a*vnoise(x,y);x*=2.03;y*=2.03;a*=.5;}return s;}
function loadImg(src){return new Promise(r=>{const i=new Image();i.onload=()=>r(i);i.src=src;});}
function grain(ctx,W,H,amt,seed){const id=ctx.getImageData(0,0,W,H),d=id.data;const R=mulberry(seed);
 for(let i=0;i<d.length;i+=4){const n=(R()-.5)*amt;d[i]+=n;d[i+1]+=n;d[i+2]+=n;}ctx.putImageData(id,0,0);}
function vignette(ctx,W,H,s){const g=ctx.createRadialGradient(W/2,H/2,H*.35,W/2,H/2,H*1.0);g.addColorStop(0,'rgba(0,0,0,0)');g.addColorStop(1,`rgba(0,0,0,${s})`);ctx.fillStyle=g;ctx.fillRect(0,0,W,H);}
function bloom(ctx,W,H,thresh,strength,blur){ // copy bright parts, downscale, blur, add
 const s=document.createElement('canvas');s.width=W/4;s.height=H/4;const c=s.getContext('2d');
 c.drawImage(ctx.canvas,0,0,W/4,H/4);const id=c.getImageData(0,0,W/4,H/4),d=id.data;
 for(let i=0;i<d.length;i+=4){const l=(d[i]*.3+d[i+1]*.59+d[i+2]*.11);const k=Math.max(0,(l-thresh)/(255-thresh));d[i]*=k;d[i+1]*=k;d[i+2]*=k;}
 c.putImageData(id,0,0);const b=document.createElement('canvas');b.width=W/4;b.height=H/4;const bc=b.getContext('2d');bc.filter=`blur(${blur}px)`;bc.drawImage(s,0,0);
 ctx.save();ctx.globalCompositeOperation='screen';ctx.globalAlpha=strength;ctx.drawImage(b,0,0,W,H);ctx.restore();}
