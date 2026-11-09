"""把与用户的沟通记录导出为 crew/CONVERSATION.md（autosync.sh 每次同步前自动运行）。
第一段会话（前两任）已冻结在 crew/CONVERSATION_part1.md；之后的会话从各 journal 目录读取并追加。"""
import json, glob, os, re
REDACT = re.compile(r'bsk__[A-Za-z0-9._\-]+')   # 公开仓库：任何 Skills Key 一律打码
HERE = os.path.dirname(os.path.abspath(__file__))
JS = ['/workspace/projects/bb073534-bbd7-4972-8ee7-8269af99396f/.hark/journal']   # 第七任（2026-11-09 10:44 起，新沙箱）；前六任已冻结在 CONVERSATION_part5.md
rows = []
for J in JS:
    for f in sorted(glob.glob(J + '/*.jsonl')):
        for l in open(f, encoding='utf-8'):
            try: rows.append(json.loads(l))
            except Exception: pass
out = [open(os.path.join(HERE, 'CONVERSATION_part5.md'), encoding='utf-8').read().rstrip(), '', '---', '', '## 第七任接手（2026-11-09 10:44 起，新沙箱）', '']
for r in rows:
    if r.get('role') not in ('user', 'assistant'): continue
    who = '用户' if r.get('role') == 'user' else 'Hark'
    out += [f'### {who} · {r.get("time","")}', '', REDACT.sub('bsk__***已打码***', r.get('text', '').strip()), '']
open(os.path.join(HERE, 'CONVERSATION.md'), 'w', encoding='utf-8').write('\n'.join(out))
print('exported', len(rows))
