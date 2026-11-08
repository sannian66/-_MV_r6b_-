"""把与用户的沟通记录导出为 crew/CONVERSATION.md（autosync.sh 每次同步前自动运行）。"""
import json, glob, os
J = '/workspace/.hark/journal/48161188-c1a2-483c-a894-09e5e091db89'
rows = []
for f in sorted(glob.glob(J + '/*.jsonl')):
    for l in open(f, encoding='utf-8'):
        try: rows.append(json.loads(l))
        except Exception: pass
start = next((i for i, r in enumerate(rows) if r.get('role') == 'user' and '接手任务' in r.get('text', '')), 0)
out = ['# 《谁定正邪》MV 沟通记录', '', '用户与 Hark（制片/总导演）的对话，自动从会话导出。最早一条用户消息内含与上一任 agent 的对话记录。', '']
for r in rows[start:]:
    who = '用户' if r.get('role') == 'user' else 'Hark'
    out += [f'### {who} · {r.get("time","")}', '', r.get('text', '').strip(), '']
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CONVERSATION.md'), 'w', encoding='utf-8').write('\n'.join(out))
print('exported', len(rows) - start)
