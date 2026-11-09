"""E 组（副歌终 + 尾奏）新精灵生图：RH_image2 4K 16:9，参考 男主_鬼屠.webp，晨光金色逆光（与破晓场景同光向同色温）。
  python3 tools/e_gen.py quote
  python3 tools/e_gen.py <job,job|all> [try]
候选落仓库外 ../gen_cands/E/；挑选后 recut_flat 抠图到 assets/gen/<镜号>/<姿势>_raw.png / _cut.png。日志追加 crew/genlog.md。"""
import sys, os, json, time, threading
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/tools')
import nb
CAND = os.path.join(os.path.dirname(os.path.realpath(ROOT)), 'gen_cands', 'E')
REF = json.load(open(ROOT + '/assets/characters/asset_urls.json'))['男主_鬼屠.webp']
CHAR = ("The same young man as in the reference image: identical face and identical hairstyle. High ponytail tied on top of the head "
        "with long messy black strands, center-parted messy bangs falling over the forehead and between the eyes, amber eyes, "
        "pale lean face, sharp straight brows, sharp jawline. Costume: black layered wuxia robe, dark crimson inner collar, wide dark "
        "leather waist sash with a dark red sash cloth, dark leather bracers on the forearms. Weapon: long single-edged Chinese dao with a black cord-wrapped hilt. ")
BG = ("Background: plain seamless flat neutral mid-gray studio backdrop, completely empty, no objects, no ground, no smoke, no sun, "
      "no text, no watermark. Lighting: warm golden sunrise backlight from behind and slightly to the right creating a bright warm golden rim "
      "outlining the silhouette, hair strands and robe edges, soft warm low fill from the front, cinematic realistic film still, photorealistic, high detail, sharp focus. ")
FULL = "The whole figure is fully inside the frame with generous empty margin on all sides; the top of the ponytail, both feet and the whole sword including its tip are fully visible and not cropped. "
JOBS = {
 'S118/xslash': "Medium-wide shot, full body, front three-quarter view, he is mid-way through a powerful diagonal slash from upper left down to lower right, "
                "the dao swung across his body with both hands, blade blurred slightly at the tip, torso twisted, front knee bent, robe hem and ponytail flying hard from the motion. Dynamic action pose. " + FULL,
 'S122/walk_toward': "Long shot, full body, seen from the front, he walks steadily straight toward the camera with a calm resolute gaze, the dao held low in his right hand "
                     "with the blade pointing down and slightly behind, left arm relaxed, robe hem and the red sash cloth blown back by wind. " + FULL,
 'S127/profile_up': "Close-up, head and shoulders, pure side profile view facing right, chin slightly raised, he looks up toward the upper right with calm resolve, "
                    "messy bangs and loose strands lifted by wind, the high ponytail visible above and behind the head and fully inside the frame, crimson inner collar visible. No sword in the frame.",
 'S129/coil': "Medium shot from the knees up, front three-quarter view, he coils for a final strike: the dao pulled far back behind his right shoulder with both hands on the hilt, "
              "blade pointing back and up, torso twisted, eyes fixed forward toward the camera with fierce intensity, ponytail swinging. Dynamic action pose. The whole blade is inside the frame.",
 'S130/hslash': "Wide shot, full body, front view, low wide stance right after a huge horizontal slash: his right arm extended fully out to his right side at shoulder height, "
                "the dao perfectly horizontal at arm's length, left hand open for balance, robe skirt flaring outward, ponytail whipping. Dynamic action pose. " + FULL,
}
LOG = ROOT + '/crew/genlog.md'
lk = threading.Lock()
def prompt(job): return JOBS[job] + " " + CHAR + BG
def body(job): return dict(model='RH_image2', prompt=prompt(job), ratio='16:9', size='4K', input=[REF])

def run(job, k):
    shot, pose = job.split('/')
    r = nb.generate(body(job), None)
    res = r.get('_error') or 'completed'; files = ''
    if not r.get('_error') and r.get('result_urls'):
        files = nb.download(r['result_urls'][:1], os.path.join(CAND, shot), f'{pose}_try{k}')[0]
    with lk:
        open(LOG, 'a').write(f"| {time.strftime('%m-%d %H:%M')} | {shot} | RH_image2 | 4K 16:9 | 3 | {r.get('task_id','')} | {res} | "
                             f"gen_cands/E/{shot}/{os.path.basename(files) if files else '-'} | [E组] {pose} try{k}: {JOBS[job][:60]} |\n")
    print(job, k, res, files, flush=True)
    return job, k, res, files

if __name__ == '__main__':
    if sys.argv[1] == 'quote':
        for j in JOBS: print(j, nb.api('POST', '/api/skills/images/quote', body(j))[1]); break
        sys.exit()
    jobs = list(JOBS) if sys.argv[1] == 'all' else sys.argv[1].split(',')
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    with ThreadPoolExecutor(5) as ex: list(ex.map(lambda j: run(j, k), jobs))
