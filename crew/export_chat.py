"""把与用户的沟通记录导出为 crew/CONVERSATION.md（autosync.sh 每次同步前自动运行）。
第一段会话（前两任）已冻结在 crew/CONVERSATION_part1.md；之后的会话从各 journal 目录读取并追加。"""
import json, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
JS = ['/workspace/projects/056c9682-39b0-4dca-a3b4-ab19434e714b/.hark/journal']   # 2026-11-09 03:20 起的接手会话
rows = []
for J in JS:
    for f in sorted(glob.glob(J + '/*.jsonl')):
        for l in open(f, encoding='utf-8'):
            try: rows.append(json.loads(l))
            except Exception: pass
out = [open(os.path.join(HERE, 'CONVERSATION_part1.md'), encoding='utf-8').read().rstrip(), '', '---', '', '## 第三任接手（2026-11-09 03:20 起）', '']
for r in rows:
    if r.get('role') not in ('user', 'assistant'): continue
    who = '用户' if r.get('role') == 'user' else 'Hark'
    out += [f'### {who} · {r.get("time","")}', '', r.get('text', '').strip(), '']
open(os.path.join(HERE, 'CONVERSATION.md'), 'w', encoding='utf-8').write('\n'.join(out))
print('exported', len(rows))
