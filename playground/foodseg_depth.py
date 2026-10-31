"""Shared monocular-depth evaluation on FoodSeg103 (classical multi-cue and ml-depth-pro notebooks).

FoodSeg103 has no depth ground truth, so depth maps are scored with what the dataset does provide:
  * the segmentation masks -- food sits on a plate/table, so for a (near-)overhead photo the food
    should be nearer to the camera than the surface around it, and depth edges should follow object edges;
  * EXIF 35 mm-equivalent focal length (a few percent of images) -- checks the estimated focal length;
  * agreement with ml-depth-pro, used as a pseudo reference for the classical estimator.
All depth maps are compared at EVAL_SIDE (longest side) so cached maps stay small.
"""
import json
import math
import pickle
import time

import cv2
import numpy as np
from PIL import Image
from scipy.stats import rankdata, spearmanr

import foodseg_common as fc

DEPTH_ROOT = fc.OUT_ROOT / 'depth'
EVAL_SIDE = 384
RING_PX = 9                    # width of the surface ring around each food region (at EVAL_SIDE)
REGION_MIN_AREA = 200          # food regions smaller than this (at EVAL_SIDE) are ignored
HEIGHT_RANGE_MM = (0.0, 100.0)  # plausible height of food above the surrounding surface
FULL_FRAME_DIAG_MM = math.hypot(36.0, 24.0)


# ---------------------------------------------------------------- data
def eval_shape(h, w, side=EVAL_SIDE):
    s = side / max(h, w)
    return max(1, round(h * s)), max(1, round(w * s))


def to_eval(depth, side=EVAL_SIDE):
    """Resize a full-resolution depth map to the evaluation resolution (float32)."""
    eh, ew = eval_shape(*depth.shape, side)
    return cv2.resize(depth.astype(np.float32), (ew, eh), interpolation=cv2.INTER_AREA)


def eval_label_mask(split, sid, side=EVAL_SIDE):
    lm = fc.load_label_mask(fc.mask_path(split, sid)).astype(np.uint8)   # FoodSeg103 ids are 0..103
    eh, ew = eval_shape(*lm.shape, side)
    return cv2.resize(lm, (ew, eh), interpolation=cv2.INTER_NEAREST)


def exif_focal_px(split, sid):
    """
    Focal length in pixels from EXIF FocalLengthIn35mmFilm, or None.
    35 mm equivalence is defined on the diagonal: f_px = f35 * diag_px / 43.27 mm.
    (Only valid if the dataset image was resized, not cropped, from the camera original.)
    """
    try:
        img = Image.open(fc.image_path(split, sid))
        f35 = img.getexif().get_ifd(0x8769).get(41989)
        if not f35:
            return None
        w, h = img.size
        return float(f35) * math.hypot(w, h) / FULL_FRAME_DIAG_MM
    except Exception:
        return None


def exif_focals(split, ids=None):
    """{id: EXIF focal length in px} for the images of a split that have one; cached as JSON."""
    DEPTH_ROOT.mkdir(parents=True, exist_ok=True)
    p = DEPTH_ROOT / f'exif_focal_{split}.json'
    if p.exists():
        found = json.loads(p.read_text())
    else:
        found = {s: f for s in fc.read_split_ids(split) if (f := exif_focal_px(split, s))}
        p.write_text(json.dumps(found, indent=1))
    if ids is None:
        return found
    keep = set(ids)
    return {s: f for s, f in found.items() if s in keep}


def hfov_deg(f_px, width):
    return math.degrees(2 * math.atan(width / (2 * f_px)))


# ---------------------------------------------------------------- depth cache
def cache_path(name, split):
    return DEPTH_ROOT / f'{name}_{split}.pkl'


def load_depth_cache(name, split):
    """{id: dict(depth=float16 (h,w) at EVAL_SIDE, f_px, width, height, sec)}, or {} if missing."""
    p = cache_path(name, split)
    if not p.exists():
        return {}
    with open(p, 'rb') as f:
        return pickle.load(f)


def save_depth_cache(cache, name, split):
    DEPTH_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = cache_path(name, split).with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_path(name, split))


def run_depth(estimate_fn, split, ids, name=None, save_every=50, progress=None):
    """
    Run estimate_fn(rgb uint8, sample_id) -> (depth_m full-res float, f_px) over ids and cache the
    results at EVAL_SIDE. With a name the cache on disk is resumed and updated; returns the cache dict.
    """
    cache = load_depth_cache(name, split) if name else {}
    todo = [s for s in ids if s not in cache]
    it = progress(todo) if progress else todo
    for n, sid in enumerate(it, 1):
        rgb = fc.load_rgb(fc.image_path(split, sid))
        t0 = time.perf_counter()
        depth, f_px = estimate_fn(rgb, sid)
        sec = time.perf_counter() - t0
        cache[sid] = dict(depth=to_eval(depth).astype(np.float16), f_px=float(f_px),
                          width=rgb.shape[1], height=rgb.shape[0], sec=sec)
        if name and n % save_every == 0:
            save_depth_cache(cache, name, split)
    if name and todo:
        save_depth_cache(cache, name, split)
    return {s: cache[s] for s in ids if s in cache}


# ---------------------------------------------------------------- mask-based metrics
def _auroc(pos, neg):
    """P(score_pos > score_neg) via ranks (ties count half)."""
    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mask_metrics(depth, label_mask, ring_px=RING_PX, min_area=REGION_MIN_AREA):
    """
    Per-image GT-free scores of a depth map (metres, near = small) against the FoodSeg103 label mask.
      food_auroc     : AUROC of nearness (-depth) separating food pixels from non-food pixels (0.5 = chance)
      ring_order     : fraction of food regions whose median depth is nearer than the surface ring around them
      height_mm      : median over regions of (ring depth - region depth), in mm (only meaningful for metric depth)
      height_ok      : fraction of regions with height inside HEIGHT_RANGE_MM
      boundary_ratio : mean |grad log depth| on GT region boundaries / elsewhere (>1 = depth edges follow objects)
    """
    d = np.maximum(depth.astype(np.float32), 1e-4)
    food = label_mask > 0
    out = {}
    if 0 < food.sum() < food.size:
        out['food_auroc'] = _auroc(-d[food], -d[~food])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    order, heights = [], []
    for c in np.unique(label_mask):
        if c == 0:
            continue
        region = label_mask == c
        if region.sum() < min_area:
            continue
        ring = cv2.dilate(region.astype(np.uint8), kernel).astype(bool) & ~food
        if ring.sum() < min_area // 4:
            continue
        dz = float(np.median(d[ring]) - np.median(d[region]))
        order.append(dz > 0)
        heights.append(dz * 1000)
    if order:
        out['ring_order'] = float(np.mean(order))
        out['height_mm'] = float(np.median(heights))
        lo, hi = HEIGHT_RANGE_MM
        out['height_ok'] = float(np.mean([lo < h <= hi for h in heights]))
    out['n_regions'] = len(order)

    logd = np.log(d)
    gy, gx = np.gradient(logd)
    g = np.hypot(gx, gy)
    edges = cv2.morphologyEx(label_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if edges.any() and (~edges).any():
        out['boundary_ratio'] = float(g[edges].mean() / max(g[~edges].mean(), 1e-8))
    out['median_depth_m'] = float(np.median(d))
    return out


# ---------------------------------------------------------------- agreement with a reference
def agreement(pred, ref, mask=None):
    """
    Scale-invariant agreement of pred with a reference depth map (both metres). pred is median-scaled to ref.
      abs_rel : mean |s*pred - ref| / ref      delta1 : fraction with max(ratio, 1/ratio) < 1.25
      si_log  : scale-invariant log RMSE (x100) spearman : rank correlation (ordering only)
    """
    p = np.maximum(pred.astype(np.float32), 1e-4)
    r = np.maximum(ref.astype(np.float32), 1e-4)
    if mask is not None:
        p, r = p[mask], r[mask]
    else:
        p, r = p.ravel(), r.ravel()
    if len(p) < 10:
        return {}
    p = p * (np.median(r) / np.median(p))
    ratio = np.maximum(p / r, r / p)
    e = np.log(p) - np.log(r)
    return dict(abs_rel=float(np.mean(np.abs(p - r) / r)),
                delta1=float(np.mean(ratio < 1.25)),
                si_log=float(np.sqrt(max(np.mean(e ** 2) - np.mean(e) ** 2, 0)) * 100),
                spearman=float(spearmanr(p, r)[0]))


# ---------------------------------------------------------------- split-level evaluation
def evaluate(cache, split, ref_cache=None, exif=None):
    """
    Average mask_metrics over the cache (and agreement with ref_cache on shared ids, all pixels and food only).
    exif: {id: f_px} -- focal-length error of the cached f_px on the images that have EXIF.
    Returns (summary dict, per-image list of dicts).
    """
    rows = []
    for sid, c in cache.items():
        lm = eval_label_mask(split, sid)
        d = c['depth'].astype(np.float32)
        if d.shape != lm.shape:
            d = cv2.resize(d, lm.shape[::-1], interpolation=cv2.INTER_LINEAR)
        row = dict(id=sid, sec=c['sec'], f_px=c['f_px'], hfov_deg=hfov_deg(c['f_px'], c['width']),
                   **mask_metrics(d, lm))
        if ref_cache is not None and sid in ref_cache:
            r = ref_cache[sid]['depth'].astype(np.float32)
            r = cv2.resize(r, lm.shape[::-1], interpolation=cv2.INTER_LINEAR) if r.shape != lm.shape else r
            row.update({f'ref_{k}': v for k, v in agreement(d, r).items()})
            row.update({f'ref_food_{k}': v for k, v in agreement(d, r, lm > 0).items()})
        if exif and sid in exif:
            row['focal_rel_err'] = abs(c['f_px'] - exif[sid]) / exif[sid]
        rows.append(row)

    summary = {}
    for k in dict.fromkeys(k for r in rows for k in r if k != 'id'):
        vals = [r[k] for r in rows if k in r and r[k] is not None and np.isfinite(r[k])]
        if vals:
            summary[k] = float(np.median(vals) if k in ('height_mm', 'median_depth_m', 'sec') else np.mean(vals))
    summary['n_images'] = len(rows)
    summary['n_focal'] = sum('focal_rel_err' in r for r in rows)
    return summary, rows


# ---------------------------------------------------------------- results
def save_result(name, split, summary, extra=None):
    """Write outputs/depth/{name}_{split}.json (read back by the comparison in the Depth Pro notebook)."""
    DEPTH_ROOT.mkdir(parents=True, exist_ok=True)
    out = dict(estimator=name, split=split, **summary, **(extra or {}))
    (DEPTH_ROOT / f'{name}_{split}.json').write_text(json.dumps(out, indent=2))
    return out


def load_results(split='test'):
    out = {}
    for p in sorted(DEPTH_ROOT.glob(f'*_{split}.json')):
        r = json.loads(p.read_text())
        if isinstance(r, dict) and 'estimator' in r:      # skips the EXIF focal-length cache
            out[r['estimator']] = r
    return out


# ---------------------------------------------------------------- visualisation / geometry
def colourise(depth, lo=None, hi=None):
    """RGB INFERNO visualisation, yellow = near, dark = far."""
    lo = np.percentile(depth, 2) if lo is None else lo
    hi = np.percentile(depth, 98) if hi is None else hi
    n = np.clip((depth - lo) / max(hi - lo, 1e-8), 0, 1)
    return cv2.cvtColor(cv2.applyColorMap(((1 - n) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
                        cv2.COLOR_BGR2RGB)


def backproject(depth, f_px, mask=None, stride=4):
    """Pinhole back-projection of depth pixels to (N,3) XYZ points; principal point at the image centre."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[vs, us].astype(np.float32)
    valid = z > 0 if mask is None else (z > 0) & mask[vs, us]
    u, v, z = us[valid], vs[valid], z[valid]
    return np.stack([(u - w / 2) * z / f_px, (v - h / 2) * z / f_px, z], axis=1)
