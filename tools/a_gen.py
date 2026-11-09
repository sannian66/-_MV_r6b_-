"""A 组（前奏 + 主歌一）新精灵生图：报价 → 提交 → 轮询 → 下载候选 → 采用。
  python3 tools/a_gen.py quote             报价全部未提交项
  python3 tools/a_gen.py submit [key ...]  提交（预算 A 组 80 点）
  python3 tools/a_gen.py poll              轮询进行中任务，完成的下载到 /workspace/work/a_cands/
  python3 tools/a_gen.py ok <key>          采用 → assets/gen/A/<key>_raw.png 并抠图 _cut.png
状态：crew/round4/A_gen_state.json；日志 crew/genlog.md。Key 由沙箱代理注入，绝不写文件。"""
import sys, os, json, time, shutil, subprocess
sys.path.insert(0, os.path.dirname(__file__))
import nb
from sprite_plan_rest import GT_LOOK, TSH_LOOK, BG, LIGHT_COLD, FRAME, HANDS
ROOT = '/workspace/work/mv'
STATE = ROOT + '/crew/round4/A_gen_state.json'
LOG = ROOT + '/crew/genlog.md'
CANDS = '/workspace/work/a_cands'
BUDGET = 80
REFS = json.load(open(ROOT + '/assets/characters/asset_urls.json'))
GT, TSH = '男主_鬼屠', '最终boss_太上皇'

JOBS = {
    'eyes': (GT, 'ECU', 'Extreme close-up horizontal strip of ONLY his eyes and brows (from mid-forehead to just below the eyes, nose bridge visible, '
                        'no mouth), loose black hair strands falling across the forehead, amber eyes wide open with a sharp cold killing stare straight into camera, '
                        'a thin cold highlight in each iris.'),
    'blade_face': (GT, 'CU', 'Close-up: he holds the dao vertically right in front of his face, the flat of the steel blade splitting the frame, '
                              'one amber eye visible on each side of the blade edge, cold stare at camera, hair strands across the face, hands on the hilt below the frame edge.'),
    'slash_up': (GT, 'LS', 'Dynamic low-angle view from below: he has just finished a rising diagonal sword cut from low left to high right, '
                            'dao held high above his right shoulder in one hand, blade pointing up to the sky, body twisted, robe and red sash flaring upward, '
                            'fierce expression looking up.'),
    'boots': (GT, 'CU', 'Close-up of ONLY his lower legs and black leather boots mid-stride stepping forward, the lower hem of the black robe and the end of the '
                         'dark red sash swinging, the black scabbard tip hanging beside the leg. NO head, NO upper body.'),
    'tsh_raise': (TSH, 'LS', 'Low-angle view: he stands tall and raises his right arm straight up high above his head, palm open facing the sky as if '
                              'summoning something down from heaven, the very wide sleeve falling down the raised arm, left arm down, cold arrogant expression, '
                              'chin raised, long hair and sleeves moving in wind.'),
    'tsh_hand': (TSH, 'CU', 'Close-up of ONLY his aged long-fingered right hand and the edge of his wide embroidered sleeve, hand descending from the top of the frame, '
                             'pressing down the square handle of a large dark jade imperial seal. NO face.'),
}


def prompt(char, size, pose):
    look = GT_LOOK if char == GT else TSH_LOOK
    return (BG + 'Subject: ' + pose + ' ' + look + ' ' + FRAME[size] + LIGHT_COLD + HANDS +
            'Photorealistic high-end wuxia film still, sharp focus, ultra detailed fabric and skin texture.')


def body(k):
    c, s, p = JOBS[k]
    return dict(model='RH_image2', prompt=prompt(c, s, p), ratio='16:9', size='4K', input=[REFS[c + '.webp']])


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {'spent': 0, 'jobs': {}}


def save(s): json.dump(s, open(STATE, 'w'), ensure_ascii=False, indent=1)


def log(k, pts, tid, res, path, note):
    open(LOG, 'a').write('| %s | A/%s | RH_image2 | 4K 16:9 | %s | %s | %s | %s | %s |\n' % (
        time.strftime('%m-%d %H:%M'), k, pts, tid, res, path, note))


def quote(k):
    st, raw = nb.api('POST', '/api/skills/images/quote', body(k))
    return json.loads(raw).get('billing', {}).get('total_points')


def submit(keys):
    s = load()
    for k in keys:
        q = quote(k)
        if q is None: print('quote fail', k); continue
        if s['spent'] + q > BUDGET: print('budget reached', s['spent']); break
        st, raw = nb.api('POST', '/api/skills/generate', body(k), timeout=60)
        try: d = json.loads(raw)
        except Exception: d = {}
        tid = d.get('task_id') or d.get('skill_task_id') or (d.get('task') or {}).get('id')
        if not tid:
            log(k, 0, '-', 'submit_failed http_%s' % st, '', 'A组'); print('fail', k, raw[:200]); continue
        s['spent'] += q
        s['jobs'].setdefault(k, []).append(dict(task_id=tid, status='running', pts=q, file=None))
        log(k, q, tid, 'submitted', '-', 'A组 %s try%d' % (k, len(s['jobs'][k])))
        print('submitted', k, tid, q)
        save(s)
    print('spent', s['spent'])


def poll():
    s = load()
    for k, tries in s['jobs'].items():
        for tr in tries:
            if tr['status'] != 'running': continue
            st, raw = nb.api('GET', '/api/skills/tasks/%s' % tr['task_id'])
            task = (json.loads(raw) or {}).get('task') or {}
            ss = (task.get('status') or '').lower()
            if ss == 'completed' and task.get('result_urls'):
                fs = nb.download(task['result_urls'], CANDS, k + '_' + tr['task_id'][:8])
                tr['status'] = 'done'; tr['file'] = fs[0]
                log(k, 0, tr['task_id'], 'completed', fs[0], 'A组 候选')
            elif ss in nb.FAILED:
                tr['status'] = ss; log(k, 0, tr['task_id'], ss, '-', 'A组 失败')
            print(k, tr['task_id'][:8], ss)
    save(s)


def ok(k):
    s = load(); tr = [x for x in s['jobs'][k] if x['file']][-1]
    os.makedirs(ROOT + '/assets/gen/A', exist_ok=True)
    raw = ROOT + '/assets/gen/A/%s_raw.png' % k
    from PIL import Image
    Image.open(tr['file']).convert('RGB').save(raw)
    subprocess.run(['python3', ROOT + '/tools/recut_flat.py', raw, raw.replace('_raw', '_cut'), '16'], check=True)
    tr['accepted'] = True; save(s)
    log(k, 0, tr['task_id'], '采用', 'assets/gen/A/%s_raw.png' % k, 'A组 采用')


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'quote':
        for k in JOBS: print(k, quote(k))
    elif c == 'submit': submit(sys.argv[2:] or list(JOBS))
    elif c == 'poll': poll()
    elif c == 'ok': ok(sys.argv[2])
