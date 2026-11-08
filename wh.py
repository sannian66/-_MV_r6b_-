import whisper,json
m=whisper.load_model("small")
lyr=open('/workspace/files/uploads/谁定正邪-d1f482175753.txt').read()
r=m.transcribe("song16k.wav",language="zh",word_timestamps=True,initial_prompt="长夜压在我刀锋之上 五百重甲拦不住一道寒光 血未冷 刀不放 谁人来定正邪一方",condition_on_previous_text=False)
json.dump(r,open("whisper.json","w"),ensure_ascii=False)
for s in r["segments"]: print(round(s["start"],2),round(s["end"],2),s["text"])
