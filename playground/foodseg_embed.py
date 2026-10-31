"""Shared food-region embedding evaluation for FoodSeg103 (HOG, Mini-ViT and DINOv2 notebooks).

Task: every food-class region of an annotation (area >= EMBED_MIN_AREA) is cropped to its bounding box,
background pixels are zeroed, and the crop is embedded. Labels are the FoodSeg103 class ids (1..103).
Train-split regions form the gallery, val/test-split regions are queries, and cosine-similarity KNN
measures how well each embedding separates the food classes.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import foodseg_common as fc

EMBED_ROOT = fc.OUT_ROOT / 'embed'
EMBED_MIN_AREA = 32 * 32
KS = (1, 3, 5, 10)


def category_names():
    """FoodSeg103 class names indexed by class id (0 = background)."""
    path = fc.PROJECT_ROOT / 'dataset' / 'foodseg103_original' / 'category_id.txt'
    names = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                names[int(parts[0])] = parts[1]
    return [names.get(i, str(i)) for i in range(104)]


# ---------------------------------------------------------------- regions
def region_index(split, min_area=EMBED_MIN_AREA, limit=None):
    """List of (sample_id, class_id) for every class region >= min_area. Cached as JSON per split."""
    EMBED_ROOT.mkdir(parents=True, exist_ok=True)
    cache = EMBED_ROOT / f'regions_{split}_a{min_area}.json'
    if cache.exists():
        regions = [tuple(r) for r in json.loads(cache.read_text())]
    else:
        regions = []
        for sid in fc.read_split_ids(split):
            lm = fc.load_label_mask(fc.mask_path(split, sid))
            ids, counts = np.unique(lm, return_counts=True)
            regions += [(sid, int(c)) for c, n in zip(ids, counts) if c != 0 and n >= min_area]
        cache.write_text(json.dumps(regions))
    if limit:
        ids = list(dict.fromkeys(sid for sid, _ in regions))[:limit]
        keep = set(ids)
        regions = [r for r in regions if r[0] in keep]
    return regions


def crop_region(rgb, label_mask, cls, size):
    """Bounding-box crop of one class region, background zeroed, resized to size x size. Returns (crop, mask)."""
    m = (label_mask == cls).astype(np.uint8)
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgb[y0:y1, x0:x1].copy()
    m = m[y0:y1, x0:x1]
    crop[m == 0] = 0
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return crop, m


def iter_crops(split, regions, size):
    """Yield (index, crop_rgb, crop_mask) for regions, loading each image only once."""
    by_image = {}
    for i, (sid, cls) in enumerate(regions):
        by_image.setdefault(sid, []).append((i, cls))
    for sid, items in by_image.items():
        rgb = fc.load_rgb(fc.image_path(split, sid))
        lm = fc.load_label_mask(fc.mask_path(split, sid))
        for i, cls in items:
            crop, m = crop_region(rgb, lm, cls, size)
            yield i, crop, m


def load_crops(split, regions, size, cache_name=None):
    """All crops as a uint8 array (N, size, size, 3); optionally cached as .npy (use for small sizes only)."""
    path = EMBED_ROOT / f'{cache_name}.npy' if cache_name else None
    if path is not None and path.exists():
        arr = np.load(path)
        if len(arr) == len(regions):
            return arr
    arr = np.zeros((len(regions), size, size, 3), dtype=np.uint8)
    for i, crop, _ in iter_crops(split, regions, size):
        arr[i] = crop
    if path is not None:
        EMBED_ROOT.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
    return arr


def labels_of(regions):
    return np.array([cls for _, cls in regions], dtype=np.int64)


# ---------------------------------------------------------------- KNN evaluation
def l2_normalise(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


@torch.no_grad()
def knn_search(gallery, query, k, chunk=1024):
    """Cosine-similarity top-k neighbours of every query in gallery. Returns (sims, idx), each (Nq, k)."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = torch.as_tensor(l2_normalise(gallery), device=dev)
    sims, idx = [], []
    for s in range(0, len(query), chunk):
        q = torch.as_tensor(l2_normalise(query[s:s + chunk]), device=dev)
        v, i = (q @ g.T).topk(k, dim=1)
        sims.append(v.cpu()); idx.append(i.cpu())
    return torch.cat(sims).numpy(), torch.cat(idx).numpy()


def knn_evaluate(gallery, gallery_y, query, query_y, ks=KS, top=5):
    """
    Similarity-weighted KNN vote for every K in ks, top-`top` hit rate, and class-balanced (macro) 1-NN accuracy.
    Returns (metrics dict, per-class 1-NN accuracy dict, 1-NN predictions).
    """
    kmax = max(max(ks), top)
    sims, idx = knn_search(gallery, query, kmax)
    nn_y = gallery_y[idx]                                   # (Nq, kmax)
    res = {}
    for k in ks:
        votes = np.zeros((len(query), int(gallery_y.max()) + 1), dtype=np.float32)
        np.add.at(votes, (np.arange(len(query))[:, None].repeat(k, 1), nn_y[:, :k]), sims[:, :k])
        res[f'knn{k}_acc'] = float((votes.argmax(1) == query_y).mean())
    res[f'top{top}_hit'] = float((nn_y[:, :top] == query_y[:, None]).any(1).mean())

    pred1 = nn_y[:, 0]
    per_class = {}
    for c in np.unique(query_y):
        sel = query_y == c
        per_class[int(c)] = dict(acc=float((pred1[sel] == c).mean()), n=int(sel.sum()))
    res['macro_acc1'] = float(np.mean([v['acc'] for v in per_class.values()]))
    res['n_gallery'] = int(len(gallery))
    res['n_query'] = int(len(query))
    return res, per_class, pred1


# ---------------------------------------------------------------- results
def save_result(name, split, metrics, extra=None):
    """Write metrics to outputs/embed/{name}_{split}.json (read back by the DINOv2 notebook's comparison)."""
    EMBED_ROOT.mkdir(parents=True, exist_ok=True)
    out = dict(embedder=name, split=split, **metrics, **(extra or {}))
    (EMBED_ROOT / f'{name}_{split}.json').write_text(json.dumps(out, indent=2))
    return out


def load_results(split='test'):
    """All saved embedder results for a split, keyed by embedder name."""
    return {json.loads(p.read_text())['embedder']: json.loads(p.read_text())
            for p in sorted(EMBED_ROOT.glob(f'*_{split}.json'))}


# ---------------------------------------------------------------- visualisation
def show_retrievals(ax_rows, query_crops, query_y, gallery_crops_fn, nn_idx, gallery_y, names, n_show=5):
    """Plot each query crop followed by its n_show nearest gallery crops (green title = correct class)."""
    for row, qc, qy, nbrs in zip(ax_rows, query_crops, query_y, nn_idx):
        row[0].imshow(qc); row[0].set_title(f'query: {names[qy]}', fontsize=8); row[0].axis('off')
        for ax, j in zip(row[1:], nbrs[:n_show]):
            ax.imshow(gallery_crops_fn(j))
            ax.set_title(names[gallery_y[j]], fontsize=8, color='green' if gallery_y[j] == qy else 'red')
            ax.axis('off')


def pca_2d(x):
    """Project rows of x onto their first two principal components (NumPy SVD, no sklearn)."""
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T
