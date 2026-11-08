import librosa,numpy as np,json
y,sr=librosa.load("song.mp3",sr=22050,mono=True)
tempo,beats=librosa.beat.beat_track(y=y,sr=sr,units='time')
rms=librosa.feature.rms(y=y,hop_length=2205)[0]  # 0.1s
t=np.arange(len(rms))*0.1
on=librosa.onset.onset_strength(y=y,sr=sr)
print("tempo",tempo,"nbeats",len(beats),"first beats",np.round(beats[:12],2))
# smoothed loudness per 2s
db=20*np.log10(rms+1e-6)
for i in range(0,len(db),20): print(f"{t[i]:6.1f}s {db[i:i+20].mean():6.1f}dB "+"#"*int(max(0,db[i:i+20].mean()+45)))
# novelty segmentation
S=librosa.feature.chroma_cqt(y=y,sr=sr); M=librosa.feature.mfcc(y=y,sr=sr)
F=np.vstack([librosa.util.normalize(S),librosa.util.normalize(M)])
b=librosa.segment.agglomerative(F,14); print("bounds",np.round(librosa.frames_to_time(b,sr=sr),1))
json.dump({"tempo":float(np.atleast_1d(tempo)[0]),"beats":[float(x) for x in beats],"rms":rms.tolist()},open("audio.json","w"))
