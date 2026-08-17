"""Film two things: a probe particle crossing the anomaly, and
primordial soup evolving under the three-line rule.

The camera is pointed at the screen — it records frames, nothing else.
The anomaly boxes drawn mid-film use only coordinates the physicist
derived from tracking (the vanish/reappear endpoints).
"""
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, '.')
from run import make_lab
from physicist.experiments import _clusters, _tordist, capture_from_debris, test_isotropy

SCALE = 4
BG, TRAIL_DECAY = 10, 0.86


def record(u, ticks, every=2):
    frames = []
    for t in range(ticks):
        if t % every == 0:
            frames.append(u.frame().copy())
        u.tick()
    return frames


def film(frames, path, boxes=(), boxes_from=0, ms=50):
    n = frames[0].shape[0]
    trail = np.zeros((n, n), dtype=float)
    imgs = []
    for i, f in enumerate(frames):
        trail = np.maximum(trail * TRAIL_DECAY, f * 255.0)
        img = np.maximum(trail, BG)
        if i >= boxes_from:
            for (y0, y1, x0, x1) in boxes:
                for yy in (y0, y1):
                    img[yy % n, x0:x1] = np.maximum(img[yy % n, x0:x1], 80)
                for xx in (x0, x1):
                    img[y0:y1, xx % n] = np.maximum(img[y0:y1, xx % n], 80)
        big = np.kron(img, np.ones((SCALE, SCALE))).astype(np.uint8)
        imgs.append(Image.fromarray(big, mode='L'))
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=ms, loop=0)
    print(f'wrote {path} ({len(imgs)} frames)')


lab = make_lab()

# --- film 1: the probe that broke locality -------------------------------
species = capture_from_debris(lab)
variants = test_isotropy(lab, species[0])
up = next(v for v in variants if v['velocity'] == (-2, 0))

u = lab()
u.poke(65, 17, up['seed'])
frames = record(u, 230, every=2)

# the physicist's sky chart: boxes around the vanish/reappear endpoints
# measured in the survey: (28,18),(14,18) and (106,34),(96,34)
boxes = [(12, 30, 12, 30), (92, 110, 28, 46)]
film(frames, 'films/anomaly.gif', boxes=boxes, boxes_from=39, ms=60)

# --- film 2: primordial soup under a three-line rule ---------------------
rng = np.random.default_rng(11)
u = lab()
n = u.frame().shape[0]
u.poke(n // 4, n // 4, (rng.random((n // 2, n // 2)) < 0.35).astype(np.uint8))
frames = record(u, 360, every=2)
film(frames, 'films/soup.gif', ms=45)
