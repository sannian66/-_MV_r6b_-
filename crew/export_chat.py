"""把与用户的沟通记录导出为 crew/CONVERSATION.md（autosync.sh 每次同步前自动运行）。
第一段会话（前两任）已冻结在 crew/CONVERSATION_part1.md；之后的会话从各 journal 目录读取并追加。"""
import json, glob, os, re
REDACT = re.compile(r'bsk__[A-Za-z0-9._\-]+')   # 公开仓库：任何 Skills Key 一律打码
HERE = os.path.dirname(os.path.abspath(__file__))
JS = ['/workspace/projects/614ceffe-275f-4d89-90f0-6cbd68b8bafd/.hark/journal']   # 2026-11-09 07:40 起的第四任接手会话（前三任已冻结在 CONVERSATION_part2.md）
rows = []
for J in JS:
    for f in sorted(glob.glob(J + '/*.jsonl')):
        for l in open(f, encoding='utf-8'):
            try: rows.append(json.loads(l))
            except Exception: pass
out = [open(os.path.join(HERE, 'CONVERSATION_part2.md'), encoding='utf-8').read().rstrip(), '', '---', '', '## 第四任接手（2026-11-09 07:40 起）', '']
for r in rows:
    if r.get('role') not in ('user', 'assistant'): continue
    who = '用户' if r.get('role') == 'user' else 'Hark'
    out += [f'### {who} · {r.get("time","")}', '', REDACT.sub('bsk__***已打码***', r.get('text', '').strip()), '']
open(os.path.join(HERE, 'CONVERSATION.md'), 'w', encoding='utf-8').write('\n'.join(out))
print('exported', len(rows))
