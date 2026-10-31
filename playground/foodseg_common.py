"""Shared dataset access and evaluation for FoodSeg103 (Mask R-CNN and SAM2 notebooks).

Predictions from any model are stored in one format so they can be scored the
same way::

    preds = {sample_id: dict(h=H, w=W, rle=[coco_rle, ...], scores=np.ndarray)}
"""
import contextlib
import io
import pickle
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# paths (independent of the notebook working directory)
PLAYGROUND = Path(__file__).resolve().parent
PROJECT_ROOT = PLAYGROUND.parent
DATA_ROOT = PROJECT_ROOT / 'dataset' / 'foodseg103'
MODELS_DIR = PLAYGROUND / 'models'
OUT_ROOT = PLAYGROUND / 'outputs'

SPLITS = ('train', 'val', 'test')
MIN_AREA = 100


# ---------------------------------------------------------------- dataset
def read_split_ids(split):
    """Return sample ids (file stems, e.g. '00006130') listed in ImageSets/{split}.txt."""
    split_file = DATA_ROOT / 'ImageSets' / f'{split}.txt'
    with open(split_file, 'r', encoding='utf-8') as f:
        return [Path(line.strip()).stem for line in f if line.strip()]


def image_path(split, sample_id):
    """Resolve the image file for a sample id inside img_dir/{split}."""
    root = DATA_ROOT / 'Images' / 'img_dir' / split
    for ext in ('.jpg', '.png', '.jpeg'):
        p = root / f'{sample_id}{ext}'
        if p.exists():
            return p
    raise FileNotFoundError(f'Image not found for {split}/{sample_id}')


def mask_path(split, sample_id):
    """Resolve the annotation PNG for a sample id inside ann_dir/{split}."""
    p = DATA_ROOT / 'Images' / 'ann_dir' / split / f'{sample_id}.png'
    if not p.exists():
        raise FileNotFoundError(f'Mask not found: {p}')
    return p


class FoodImagePreprocessor:
    """Specialized preprocessing for food images."""

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def enhance_food_image(self, image):
        """Enhance the given food image using CLAHE and color adjustments."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return self.adjust_food_colors(enhanced)

    def adjust_food_colors(self, image):
        """Enhance colors typical in food photography."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


_preprocessor = None


def load_rgb(path, preprocess=False):
    """Read an image as RGB uint8, optionally applying FoodImagePreprocessor."""
    global _preprocessor
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if preprocess:
        if _preprocessor is None:
            _preprocessor = FoodImagePreprocessor()
        rgb = _preprocessor.enhance_food_image(rgb)
    return rgb


def load_label_mask(path):
    """Read an annotation PNG as an (H, W) int32 array of class ids."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f'Could not read mask: {path}')
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int32)


def gt_instances(label_mask, min_area=MIN_AREA):
    """One boolean mask per non-zero class region with area >= min_area. Returns (N, H, W)."""
    masks = []
    for cls_id in np.unique(label_mask):
        if int(cls_id) == 0:
            continue
        m = label_mask == cls_id
        if int(m.sum()) >= min_area:
            masks.append(m)
    if not masks:
        return np.zeros((0, *label_mask.shape), dtype=bool)
    return np.stack(masks)


# ---------------------------------------------------------------- prediction cache
def encode_masks(masks):
    """(N, H, W) bool/uint8 array -> list of COCO RLE dicts."""
    masks = np.asarray(masks)
    if masks.shape[0] == 0:
        return []
    return mask_utils.encode(np.asfortranarray(masks.transpose(1, 2, 0).astype(np.uint8)))


def save_preds(preds, path, meta=None):
    """Pickle a prediction cache (plus optional metadata such as config and timing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(dict(preds=preds, meta=meta or {}), f)


def load_preds(path):
    """Load a prediction cache. Returns (preds, meta)."""
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj['preds'], obj['meta']


@lru_cache(maxsize=None)
def _gt_split(split, min_area):
    """Cache of GT RLEs per sample id for a split: id -> (rles, h, w)."""
    out = {}
    for sid in read_split_ids(split):
        lm = load_label_mask(mask_path(split, sid))
        out[sid] = (encode_masks(gt_instances(lm, min_area)), *lm.shape)
    return out


def gt_for(split, sample_id, min_area=MIN_AREA):
    """GT RLEs, height and width for one sample."""
    return _gt_split(split, min_area)[sample_id]


# ---------------------------------------------------------------- matching metrics
def build_iou_cache(preds, split, min_area=MIN_AREA):
    """Per-image prediction/GT IoU arrays in the format used by metrics()."""
    cache = []
    for sid, p in preds.items():
        gt_rles, _, _ = gt_for(split, sid, min_area)
        scores = np.asarray(p['scores'], dtype=np.float32)
        if len(p['rle']) == 0 or len(gt_rles) == 0:
            iou = np.zeros((len(p['rle']), len(gt_rles)))
        else:
            iou = np.asarray(mask_utils.iou(list(p['rle']), list(gt_rles), [0] * len(gt_rles)))
        cache.append(dict(id=sid, scores=scores, iou=iou, n_gt=len(gt_rles)))
    return cache


def _greedy_match(iou, thr):
    """One-to-one greedy assignment; returns list of (pred_idx, gt_idx) pairs."""
    if iou.size == 0:
        return []
    ii, jj = np.nonzero(iou >= thr)
    used_p, used_g, pairs = set(), set(), []
    for k in np.argsort(-iou[ii, jj]):
        i, j = int(ii[k]), int(jj[k])
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        pairs.append((i, j))
    return pairs


def metrics(cache, score_thr, iou_thrs=(0.5, 0.75)):
    """Compute precision, recall, F1 at each IoU threshold."""
    stats = {t: dict(tp=0, fp=0, fn=0) for t in iou_thrs}
    for c in cache:
        keep = c['scores'] >= score_thr
        iou = c['iou'][keep]
        for t in iou_thrs:
            pairs = _greedy_match(iou, t)
            s = stats[t]
            s['tp'] += len(pairs)
            s['fp'] += iou.shape[0] - len(pairs)
            s['fn'] += c['n_gt'] - len(pairs)
    res = {}
    for t in iou_thrs:
        n = int(round(t * 100))
        s = stats[t]
        p = s['tp'] / max(1, s['tp'] + s['fp'])
        r = s['tp'] / max(1, s['tp'] + s['fn'])
        f = 2 * p * r / max(1e-9, p + r)
        res[f'precision@{n}'] = p
        res[f'recall@{n}'] = r
        res[f'f1@{n}'] = f
    return res


def per_image_f1(cache, score_thr, iou_thr=0.5):
    """F1 per image (dict id -> F1) at one IoU threshold."""
    out = {}
    for c in cache:
        iou = c['iou'][c['scores'] >= score_thr]
        tp = len(_greedy_match(iou, iou_thr))
        fp, fn = iou.shape[0] - tp, c['n_gt'] - tp
        out[c['id']] = 2 * tp / max(1, 2 * tp + fp + fn)
    return out


def gt_coverage(cache):
    """Best IoU achieved by any prediction for every GT instance (no score threshold)."""
    best = []
    for c in cache:
        if c['n_gt'] == 0:
            continue
        if c['iou'].shape[0] == 0:
            best.extend([0.0] * c['n_gt'])
        else:
            best.extend(c['iou'].max(axis=0).tolist())
    return np.asarray(best)


def threshold_sweep(cache, score_grid):
    """metrics() for every score threshold in score_grid, as a list of dicts."""
    return [dict(score_thr=round(float(t), 3), **metrics(cache, float(t))) for t in score_grid]


# ---------------------------------------------------------------- COCO mask AP
def coco_ap(preds, split, min_area=MIN_AREA):
    """Class-agnostic COCO segm AP/AR over the images in preds."""
    images, gt_anns, dt_anns = [], [], []
    for img_id, (sid, p) in enumerate(preds.items(), start=1):
        gt_rles, h, w = gt_for(split, sid, min_area)
        images.append(dict(id=img_id, height=h, width=w))
        for rle in gt_rles:
            gt_anns.append(dict(id=len(gt_anns) + 1, image_id=img_id, category_id=1, iscrowd=0,
                                segmentation=rle, area=float(mask_utils.area(rle)),
                                bbox=mask_utils.toBbox(rle).tolist()))
        for rle, s in zip(p['rle'], p['scores']):
            dt_anns.append(dict(image_id=img_id, category_id=1, segmentation=rle, score=float(s)))

    keys = ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'AR1', 'AR10', 'AR100')
    if not gt_anns or not dt_anns:
        return {k: 0.0 for k in keys}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = dict(images=images, annotations=gt_anns, categories=[dict(id=1, name='food')])
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt_anns)
        ev = COCOeval(coco_gt, coco_dt, iouType='segm')
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {k: float(v) for k, v in zip(keys, ev.stats[:9])}


# ---------------------------------------------------------------- semantic foreground IoU
def fg_iou(preds, split, score_thr, min_area=MIN_AREA):
    """Binary foreground IoU: union of kept predictions vs union of GT instances."""
    per_img, inter_sum, union_sum = [], 0, 0
    for sid, p in preds.items():
        gt_rles, h, w = gt_for(split, sid, min_area)
        gt_fg = mask_utils.decode(mask_utils.merge(list(gt_rles))).astype(bool) if gt_rles else np.zeros((h, w), bool)
        kept = [r for r, s in zip(p['rle'], p['scores']) if s >= score_thr]
        pr_fg = mask_utils.decode(mask_utils.merge(kept)).astype(bool) if kept else np.zeros((h, w), bool)
        inter = int(np.logical_and(gt_fg, pr_fg).sum())
        union = int(np.logical_or(gt_fg, pr_fg).sum())
        inter_sum += inter; union_sum += union
        if union > 0:
            per_img.append(inter / union)
    return dict(fg_iou_mean=float(np.mean(per_img)) if per_img else 0.0,
                fg_iou_pixel=inter_sum / max(1, union_sum))


# ---------------------------------------------------------------- combined
def evaluate(preds, split, score_thr, min_area=MIN_AREA):
    """All metrics for one prediction cache at the chosen score threshold."""
    cache = build_iou_cache(preds, split, min_area)
    cov = gt_coverage(cache)
    n_kept = [int((np.asarray(p['scores']) >= score_thr).sum()) for p in preds.values()]
    return dict(
        n_images=len(preds),
        score_thr=score_thr,
        **coco_ap(preds, split, min_area),
        **metrics(cache, score_thr),
        gt_coverage_mean=float(cov.mean()) if cov.size else 0.0,
        gt_coverage_at50=float((cov >= 0.5).mean()) if cov.size else 0.0,
        **fg_iou(preds, split, score_thr, min_area),
        preds_per_image=float(np.mean(n_kept)) if n_kept else 0.0,
    )


# ---------------------------------------------------------------- visualisation
PALETTE = np.array([[230, 25, 75], [60, 180, 75], [0, 130, 200],
                    [245, 130, 48], [145, 30, 180], [70, 200, 200],
                    [240, 50, 230], [210, 245, 60], [250, 190, 212],
                    [0, 128, 128], [170, 110, 40], [128, 0, 0]], dtype=np.uint8)


def overlay(img, masks, alpha=0.6):
    """Blend a list/array of boolean masks onto an RGB uint8 image."""
    out = img.copy()
    for i, m in enumerate(masks):
        m = np.asarray(m, dtype=bool)
        colour = PALETTE[i % len(PALETTE)]
        out[m] = (out[m] * (1 - alpha) + colour * alpha).astype(np.uint8)
    return out


def decode_kept(p, score_thr, max_masks=None):
    """Decode a cached prediction's masks with score >= score_thr (highest score first)."""
    order = np.argsort(-np.asarray(p['scores']))
    rles = [p['rle'][i] for i in order if p['scores'][i] >= score_thr][:max_masks]
    return [mask_utils.decode(r).astype(bool) for r in rles]
