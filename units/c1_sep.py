"""副歌一鼓轨分离：htdemucs 只跑 56–92s → out/c1_drums.wav（hitmap.py 优先用它）"""
import torch, soundfile as sf, numpy as np, subprocess
from demucs.pretrained import get_model
from demucs.apply import apply_model
R='/workspace/work/mv'
subprocess.run(['ffmpeg','-y','-v','error','-ss','56','-t','36','-i',R+'/song.mp3','-ar','44100','-ac','2','/tmp/c1.wav'],check=True)
wav,sr=sf.read('/tmp/c1.wav',dtype='float32'); wav=torch.tensor(wav.T)
torch.set_num_threads(2)
m=get_model('htdemucs'); m.eval()
ref=wav.mean(0); w=(wav-ref.mean())/ref.std()
with torch.no_grad(): src=apply_model(m,w[None],split=True,overlap=0.25,progress=False)[0]
src=src*ref.std()+ref.mean()
for name in ('drums','bass'):
    sf.write(f'{R}/out/c1_{name}.wav',src[m.sources.index(name)].numpy().T,sr)
print('ok')
