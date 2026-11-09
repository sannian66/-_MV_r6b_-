"""把与用户的沟通记录导出为 crew/CONVERSATION.md（autosync.sh 每次同步前自动运行）。
第一段会话（前两任）已冻结在 crew/CONVERSATION_part1.md；之后的会话从各 journal 目录读取并追加。"""
import json, glob, os, re
REDACT = re.compile(r'bsk__[A-Za-z0-9._\-]+')   # 公开仓库：任何 Skills Key 一律打码
HERE = os.path.dirname(os.path.abspath(__file__))
JS = ['/workspace/projects/5196904f-65f8-48f0-848c-4c03aa77b35b/.hark/journal']   # 2026-11-09 09:58 起的第六任接手会话（前五任已冻结在 CONVERSATION_part4.md）
rows = []
for J in JS:
    for f in sorted(glob.glob(J + '/*.jsonl')):
        for l in open(f, encoding='utf-8'):
            try: rows.append(json.loads(l))
            except Exception: pass
out = [open(os.path.join(HERE, 'CONVERSATION_part4.md'), encoding='utf-8').read().rstrip(), '', '---', '', '## 第六任接手（2026-11-09 09:58 起，新沙箱）', '']
for r in rows:
    if r.get('role') not in ('user', 'assistant'): continue
    who = '用户' if r.get('role') == 'user' else 'Hark'
    out += [f'### {who} · {r.get("time","")}', '', REDACT.sub('bsk__***已打码***', r.get('text', '').strip()), '']
open(os.path.join(HERE, 'CONVERSATION.md'), 'w', encoding='utf-8').write('\n'.join(out))
print('exported', len(rows))
