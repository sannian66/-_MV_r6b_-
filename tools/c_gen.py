"""C 组（预副歌二 + 副歌二）新精灵生图：报价 → 提交 → 轮询（分离，适配无后台沙箱）。上限 80 积分。
  python3 tools/c_gen.py quote               报价（不建任务）
  python3 tools/c_gen.py submit [名字,...]   提交（默认全部未提交）
  python3 tools/c_gen.py poll                轮询进行中任务，完成的下载到 ../c_cands/
  python3 tools/c_gen.py status
状态 crew/round4/C_gen_state.json；日志 crew/genlog.md。Key 由沙箱代理注入，绝不写文件。"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import nb
from sprite_plan_rest import GT_LOOK, TSH_LOOK, BG, HANDS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = ROOT + '/crew/round4/C_gen_state.json'
CANDS = os.path.join(os.path.dirname(ROOT), 'c_cands')
LOG = ROOT + '/crew/genlog.md'
BUDGET = 80
REFS = json.load(open(ROOT + '/assets/characters/asset_urls.json'))
STORM = ('Lighting: harsh cold white lightning flash from behind and above as a hard rim light outlining hair, shoulders and sword, '
         'very dim cold fill from the front, deep shadows, high contrast, wet glossy robe, cinematic photorealistic film still. ')
FULL = 'Framing: full body, figure about 70% of frame height, centered, generous margin on all sides, nothing cropped: top of ponytail, sleeves, whole sword and both feet inside the frame. '
JOBS = {
 'slash_strings': ('男主_鬼屠', 'C09', GT_LOOK + ' Pose: low camera angle looking up at him, standing with feet apart, his right arm whipping the dao UPWARD over his head in a fast one-handed diagonal slash toward the upper left, '
                   'blade fully extended above his head, left hand open for balance, chin raised, fierce eyes looking up at what he cuts, robe and red sash swinging. ' + FULL + STORM + HANDS + BG),
 'leap_arc': ('男主_鬼屠', 'C14', GT_LOOK + ' Pose: seen from the side facing right, leaping very high in the air at the apex of a jump, body arched back like a drawn bow, both hands gripping the dao raised high behind his head ready to strike down, '
              'knees tucked, robe hem, sleeves and long ponytail streaming upward and back. ' + FULL + STORM + HANDS + BG),
 'slash_diag_down': ('男主_鬼屠', 'C19', GT_LOOK + ' Pose: three-quarter view from behind his right shoulder, deep lunge, finishing a powerful two-handed diagonal downward slash from upper right to lower left, '
                     'blade ending low near the ground on the left, body twisted, face in sharp profile, ponytail and robe whipping with the motion. ' + FULL + STORM + HANDS + BG),
 'slash_back': ('男主_鬼屠', 'C25', GT_LOOK + ' Pose: seen squarely from BEHIND, his back to the camera, wide low stance, right arm fully extended to the right holding the dao horizontally at shoulder height '
                'at the end of a huge horizontal sweeping cut, left arm out to the left, robe flaring out in a circle, ponytail swinging. ' + FULL + STORM + HANDS + BG),
 'look_up_back': ('男主_鬼屠', 'C07', GT_LOOK + ' Pose: seen from behind and slightly below, standing still, head tilted far back looking straight up at the sky, the dao hanging low in his right hand pointing at the ground, '
                  'shoulders squared, wet hair and ponytail heavy, robe hem moving in wind. ' + FULL + STORM + HANDS + BG),
 'tsh_brush': ('最终boss_太上皇', 'C01', TSH_LOOK + ' Close-up of ONLY his right hand and wide sleeve, NO face in frame: long pale bony fingers holding a slender black Chinese writing brush tipped with vermilion red ink, '
               'brush tip pointing down as if writing, wide charcoal sleeve with faint embroidery draping, viewed from the side. Framing: close-up, hand and brush fill the middle of the frame with margin, brush tip fully visible. '
               'Lighting: warm golden candlelight from the left, deep shadows, cinematic photorealistic film still. ' + HANDS + BG),
 'blade_hot': ('男主_鬼屠', 'C11/C12', 'Macro close-up of ONLY his right hand gripping the black cord-wrapped handle of a long single-edged Chinese dao, NO face, NO head: '
               'the blade extends horizontally across the whole frame to the right, the steel glowing deep orange-red from heat along the cutting edge like freshly forged iron, '
               'dark leather wrist guard and black sleeve edge at the left, water droplets beading on the flat of the blade. Framing: the entire blade from handle to tip inside the frame with margin, blade horizontal. '
               'Lighting: the glowing edge is the key light, cold rim from above, deep shadows, cinematic photorealistic film still. ' + HANDS + BG),
 'blade_level': ('男主_鬼屠', 'C38', 'Close-up of ONLY his two hands, NO face, NO head: both hands hold a long single-edged Chinese dao perfectly level and horizontal like the beam of a scale, '
                 'left hand on the black cord-wrapped handle, right hand two fingers resting flat under the spine near the middle of the blade, black sleeves and leather wrist guards, '
                 'cold polished steel with a wavy temper line, wet with small water droplets along the blade. Framing: the whole blade from pommel to tip inside the frame with margin, blade horizontal across the frame. '
                 'Lighting: cold white lightning rim light from above, dim fill, cinematic photorealistic film still. ' + HANDS + BG),
}


def load(): return json.load(open(STATE)) if os.path.exists(STATE) else {'spent': 0, 'jobs': {}}
def save(s): json.dump(s, open(STATE, 'w'), ensure_ascii=False, indent=1)
def body(k): ch, _, pr = JOBS[k]; return dict(model='RH_image2', prompt=pr, ratio='16:9', size='4K', input=[REFS[ch + '.webp']])
def log(name, pts, tid, res, path, note):
    open(LOG, 'a').write('| %s | C组/%s | RH_image2 | 4K 16:9 | %s | %s | %s | %s | %s |\n' % (time.strftime('%m-%d %H:%M'), name, pts, tid, res, path, note))


def quote(k): return json.loads(nb.api('POST', '/api/skills/images/quote', body(k))[1]).get('billing', {}).get('total_points')


def submit(names):
    s = load()
    for k in names:
        v = s['jobs'].setdefault(k, {'tries': []})
        if v['tries'] and v['tries'][-1]['status'] == 'running': continue
        q = quote(k)
        if q is None: print('quote unavailable; stop'); break
        if s['spent'] + q > BUDGET: print('budget reached', s['spent']); break
        st, raw = nb.api('POST', '/api/skills/generate', body(k), timeout=60)
        try: d = json.loads(raw)
        except Exception: d = {}
        tid = d.get('task_id') or d.get('skill_task_id') or (d.get('task') or {}).get('id')
        n = len(v['tries']) + 1
        if not tid:
            log(k, 0, '-', 'submit_failed http_%s' % st, '', 'try%d' % n); print('fail', k, st, raw[:200]); continue
        s['spent'] += q; v['tries'].append(dict(n=n, task_id=tid, status='running', pts=q, file=None))
        log(k, q, tid, 'submitted', '-', '%s try%d (quote %s)' % (JOBS[k][1], n, q)); print('submitted', k, tid, q)
        save(s)
    save(s); print('spent', s['spent'])


def poll():
    s = load()
    for k, v in s['jobs'].items():
        tr = v['tries'][-1]
        if tr['status'] != 'running': continue
        st, raw = nb.api('GET', '/api/skills/tasks/%s' % tr['task_id'])
        try: task = json.loads(raw).get('task') or {}
        except Exception: task = {}
        sta = (task.get('status') or '').lower()
        if sta == 'completed' and task.get('result_urls'):
            f = nb.download(task['result_urls'][:1], CANDS, '%s_try%d' % (k, tr['n']))[0]
            tr['status'] = 'done'; tr['file'] = f; log(k, 0, tr['task_id'], 'completed', f, 'try%d' % tr['n']); print('done', k, f)
        elif sta in nb.FAILED:
            tr['status'] = 'failed'; log(k, 0, tr['task_id'], sta, '-', 'try%d' % tr['n']); print('failed', k, sta)
        else: print(k, sta or 'unknown')
    save(s)


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'quote':
        for k in JOBS: print(k, quote(k))
    elif c == 'submit': submit(sys.argv[2].split(',') if len(sys.argv) > 2 else [k for k in JOBS if k not in load()['jobs']])
    elif c == 'poll': poll()
    elif c == 'status': print(json.dumps(load(), ensure_ascii=False, indent=1))
