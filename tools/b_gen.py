"""[B组] 预副歌一 + 主歌二 精灵生图（第四轮，预算 ≤80 积分）。提交/轮询分离。
  python3 tools/b_gen.py quote            报价全部待提交
  python3 tools/b_gen.py submit [名字..]   报价 → 提交（记 crew/genlog.md）
  python3 tools/b_gen.py poll             轮询一次，完成的下载到 ../d_cands/
  python3 tools/b_gen.py ok <名字>         采用最新一抽 → assets/gen/B_r4/<名字>_raw.png
  python3 tools/b_gen.py bad <名字> 原因   判废（下次 submit 重抽）
状态：crew/round4/B_gen_state.json。Key 由代理注入，绝不写文件。"""
import sys, os, json, time, shutil
sys.path.insert(0, os.path.dirname(__file__))
import nb, sprite_plan_rest as plan
ROOT = plan.ROOT
STATE = os.path.join(ROOT, 'crew/round4/B_gen_state.json')
CANDS = os.path.join(os.path.dirname(ROOT), 'b_cands')
OUT = os.path.join(ROOT, 'assets/gen/B_r4')
LOG = os.path.join(ROOT, 'crew/genlog.md')
BUDGET = 80
REFS = json.load(open(os.path.join(ROOT, 'assets/characters/asset_urls.json')))
GT, TSH = plan.GT, plan.TSH
BG = plan.BG
COLD = plan.LIGHT_COLD
SOFT = 'Lighting: soft even overcast light from above-left, gentle shadows, readable folds and details in the black clothes, cinematic photorealistic film still. '

JOBS = {
 'slash_h': (GT, 'LS', 'full body facing the camera in three-quarter view, low wide stance, caught at the END of a powerful horizontal sword slash from left to right, the dao held in his right hand extended far out to the right at shoulder height, the blade a clean sharp straight horizontal line fully visible, left arm thrown back for balance, robe hem and red sash flaring, ponytail whipping, fierce determined eyes, crisp sharp image with no motion blur', COLD),
 'drag_blade': (GT, 'LS', 'full body seen from the side (profile facing right), striding forward and leaning low, dragging the long dao behind him in his right hand so the blade tip scrapes along the ground at lower left, blade fully visible, head down, ponytail streaming behind, crisp sharp image', COLD),
 'tsh_smirk': (TSH, 'CU', 'close-up of his face in three-quarter view turned slightly to the right, lit from below by warm candlelight, a cold arrogant contemptuous half smile, heavy-lidded eyes looking down at the viewer, hair strands over his shoulders, the top bun with hairpin fully visible inside the frame', plan.LIGHT_WARM),
 'tsh_brush_hand': (TSH, 'ECU', 'extreme close-up of ONLY his right hand and wide sleeve cuff: a gaunt pale old hand with long fingernails holding a thin Chinese calligraphy brush vertically with the tip pointing DOWN, the brush tip soaked in bright vermilion red ink with one red drop hanging from it, no face visible', plan.LIGHT_WARM),
 'blade_face': (GT, 'MCU', 'medium close-up, holding the dao vertically right in front of his face with both hands on the handle at chin height, the flat of the blade dividing his face down the middle so one amber eye looks past each side of the blade, intense glare at the camera, loose hair strands', COLD),
 'back_stand': (GT, 'LS', 'full body seen squarely from BEHIND, standing still with feet apart, the long dao held low in his right hand pointing down and slightly outward, head raised looking far into the distance, ponytail and red sash blown to the left by wind', COLD),

 # 第四棒：V01/V02 用真实插地刀（替掉程序灰板刀 + 十字护手）
 'blade_stuck': (None, 'LS', 'product-style shot of ONLY the hero\'s long single-edged Chinese dao saber from the reference sheet (no person, no hands, no body): the saber is planted diagonally into a small clump of dark wet mud, tilted about 15 degrees, blade tip buried, the whole blade and the complete handle visible vertically centred and filling most of the frame height, slightly low camera angle; dark forged steel with a subtle hamon line, wet surface with rain water beads and thin streams running down the flat of the blade, a round dark iron guard (tsuba) with one large clear raindrop hanging from its lower edge, handle tightly wrapped in dark red cord with a short red tassel. Only the small mud clump at the base, nothing else on the ground', COLD),

 # 第五棒（总导演第八任）：P13/P14 去横刀撞副歌一、纸人群改生图精灵、V20 重做
 'slash_up': (GT, 'LS', 'low-angle full body shot, caught at the END of a powerful RISING diagonal sword slash from lower left to upper right: the long dao held in his right hand high above his right shoulder, the blade pointing diagonally up to the upper right corner and fully visible, left arm swung down and back, body twisted, front knee bent, robe hem and red sash flaring upward, ponytail whipping, fierce eyes, crisp sharp image with no motion blur', COLD),
 'hand_hilt': (GT, 'CU', 'close-up of ONLY his right hand and forearm, no face: the hand clenched hard around the dark red cord-wrapped hilt of the dao, the round dark iron guard just above the fist, the first part of the straight steel blade rising diagonally toward the upper right and leaving the frame, black sleeve cuff and leather wrist guard, veins on the back of the hand, knuckles tight', COLD),
 'raise_blade': (GT, 'LS', 'very low-angle full body shot from below, standing with feet apart, holding the long dao straight up above his head with his right arm fully extended, the blade pointing vertically to the sky and fully visible, left hand clenched at his side, face tilted up, robe and red sash blown by strong wind, ponytail streaming', COLD),
 'look_down': (GT, 'LS', 'full body in three-quarter view from the front-left, standing still with feet apart, head bowed looking straight DOWN at the ground just in front of his feet, the long dao hanging low in his right hand with the tip almost touching the ground, shoulders slightly hunched, ponytail falling forward over one shoulder, solemn', COLD),
 'effigy': (None, 'LS', 'a single life-size traditional Chinese funeral paper effigy figure (zhizha paper man) standing upright, front view, made of pale off-white and grey mulberry paper glued over a thin bamboo frame, visible paper folds, creases, torn edges and bamboo ribs at the sleeves, a long paper robe with a faded red paper sash, arms hanging, a stiff blank painted face with two small black ink eyes and faint pink cheek circles, slightly eerie, aged and stained paper', SOFT),
 'effigy_fall': (None, 'LS', 'a single traditional Chinese funeral paper effigy figure (zhizha paper man) tumbling and falling sideways in mid-air, body tilted about 60 degrees, arms limp, made of pale off-white and grey mulberry paper over a thin bamboo frame, the paper soaked and darkened by rain with grey ink stains bleeding through, crumpled and partly torn, a long paper robe, blank painted face with small black ink eyes', SOFT),
 'crowd_bow': (None, 'LS', 'a straight row of six life-size traditional Chinese funeral paper effigy figures (zhizha paper people) in strict side profile all facing LEFT, evenly spaced and slightly overlapping, every figure bowing deeply at the waist with hands pressed together, made of pale off-white and grey mulberry paper over thin bamboo frames, long paper robes, small paper caps, blank painted faces, visible creases and torn paper edges; all six figures complete from head to feet inside the frame', SOFT),
 'crowd_back': (None, 'MS', 'a tight crowd of about eight life-size traditional Chinese funeral paper effigy figures (zhizha paper people) seen from BEHIND at shoulder height, heads bowed low, backs of paper heads with painted black hair buns and paper caps, rounded paper shoulders and long paper robes, made of off-white and grey mulberry paper over bamboo frames with creases and torn edges, arranged in two staggered rows', SOFT),
}
INPUTS = {'blade_stuck': ['男主_鬼屠.webp']}


def prompt(name):
    char, size, pose, light = JOBS[name]
    look = '' if char is None else (plan.TSH_LOOK if char == TSH else plan.GT_LOOK) + ' '
    extra = ('Keep the faces and outfits of the four reference characters, in order from left to right. ' if name == 'companions' else '')
    return pose[0].upper() + pose[1:] + '. ' + plan.FRAME.get(size, '') + plan.HANDS + look + extra + BG + light


def body(name):
    if JOBS[name][0] is None and name not in INPUTS:
        return dict(model='RH_image2', prompt=prompt(name), ratio='16:9', size='4K')
    refs = INPUTS.get(name) or [JOBS[name][0] + '.webp']
    return dict(model='RH_image2', prompt=prompt(name), ratio='16:9', size='4K', input=[REFS[r] for r in refs])


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {'spent': 0, 'jobs': {}}


def save(s):
    json.dump(s, open(STATE, 'w'), ensure_ascii=False, indent=1)


def log(name, pts, tid, res, path, note):
    open(LOG, 'a').write('| %s | B组/%s | RH_image2 | 4K 16:9 | %s | %s | %s | %s | %s |\n' % (
        time.strftime('%m-%d %H:%M'), name, pts, tid, res, path, note))


def quote(name):
    st, raw = nb.api('POST', '/api/skills/images/quote', body(name))
    return json.loads(raw).get('billing', {}).get('total_points'), st, raw[:300]


def submit(names):
    s = load()
    for n in names:
        j = s['jobs'].setdefault(n, {'tries': [], 'state': 'todo'})
        if j['state'] in ('running', 'done'): continue
        q, st, raw = quote(n)
        if q is None: print(n, 'quote failed', st, raw); continue
        if s['spent'] + q > BUDGET: print('budget stop', s['spent'], q); break
        stt, raw = nb.api('POST', '/api/skills/generate', body(n), timeout=60)
        d = json.loads(raw)
        tid = d.get('task_id') or d.get('skill_task_id') or (d.get('task') or {}).get('id')
        if not tid: print(n, 'submit failed', stt, raw[:300]); log(n, 0, '-', 'submit_failed', '-', raw[:80]); continue
        s['spent'] += q; j['tries'].append({'tid': tid, 'pts': q}); j['state'] = 'running'
        log(n, q, tid, 'submitted', '-', 'try%d %s' % (len(j['tries']), JOBS[n][2][:60]))
        print(n, 'submitted', tid, q)
    save(s); print('spent', s['spent'])


def poll():
    s = load()
    for n, j in s['jobs'].items():
        if j['state'] != 'running': continue
        tid = j['tries'][-1]['tid']
        st, raw = nb.api('GET', '/api/skills/tasks/%s' % tid, timeout=30)
        t = (json.loads(raw) or {}).get('task') or {}
        stt = (t.get('status') or '').lower()
        if stt == 'completed' and t.get('result_urls'):
            p = nb.download(t['result_urls'][:1], CANDS, '%s_try%d' % (n, len(j['tries'])))[0]
            j['state'] = 'review'; j['tries'][-1]['file'] = p
            log(n, 0, tid, 'completed', p.replace(os.path.dirname(ROOT) + '/', ''), 'try%d' % len(j['tries']))
            print(n, 'done', p)
        elif stt in nb.FAILED:
            j['state'] = 'todo'; log(n, 0, tid, stt, '-', 'failed'); print(n, stt)
        else:
            print(n, stt or st)
    save(s)


def verdict(n, ok, why=''):
    s = load(); j = s['jobs'][n]
    if ok:
        os.makedirs(OUT, exist_ok=True)
        dst = os.path.join(OUT, n + '_raw.png')
        from PIL import Image
        Image.open(j['tries'][-1]['file']).convert('RGB').save(dst)
        j['state'] = 'done'; log(n, 0, j['tries'][-1]['tid'], '采用', 'assets/gen/B_r4/%s_raw.png' % n, 'try%d 采用' % len(j['tries']))
    else:
        j['state'] = 'todo'; log(n, 0, j['tries'][-1]['tid'], '判废', '-', why)
    save(s)


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'quote':
        for n in JOBS: print(n, quote(n)[0])
    elif c == 'submit': submit(sys.argv[2:] or list(JOBS))
    elif c == 'poll': poll()
    elif c == 'ok': verdict(sys.argv[2], True)
    elif c == 'bad': verdict(sys.argv[2], False, ' '.join(sys.argv[3:]))
    elif c == 'prompt': print(prompt(sys.argv[2]))
