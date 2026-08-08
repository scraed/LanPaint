"""Per-frame video mask interpolation for the LanPaint video mask editor.

Keyframes are grayscale PNGs painted in the frontend at a capped resolution.
This module loads them, morphs between them via signed-distance-field (SDF)
level-set interpolation (at keyframe resolution), and upscales once to the
frame size.

The interpolation contract is shared with the frontend preview
(web/lanpaint_video_mask_editor.js): what the user sees is exactly what
``interpolate_masks`` produces, pixel for pixel.

Each keyframe mask is binarized at 0.5 and turned into a signed distance
field (positive inside, negative outside, 0 on the boundary). For a frame t
between keyframes k_i and k_{i+1} with w = (t - k_i) / (k_{i+1} - k_i):

    d(p)     = (1 - w) * sdf_i(p) + w * sdf_{i+1}(p)
    mask(p)  = sigmoid(d(p) / softness)        softness = 1.0 pixel

The zero level set of the blended field slides linearly between the two
shapes (translation, growth, shrink, merge/split), keeping edges sharp
instead of cross-fading them. Frames that are exact keyframes return the
original painted mask unchanged. Frames outside the keyframe window (before
the first or after the last keyframe) have NO mask (all zeros): the mask
exists only at keyframes and between them. Mask convention: 1 = regenerate,
0 = keep.
"""

import json
from typing import Dict, Tuple

import numpy as np
from PIL import Image


def load_keyframe_png(path: str, size: Tuple[int, int] | None = None) -> np.ndarray:
    """Load a mask PNG as float32 [H, W] in [0, 1].

    PNGs saved by the mask editor carry the mask in the alpha channel; a plain
    grayscale PNG (e.g. produced by other tools) is read via luminance.

    ``size`` is (width, height); the mask is resized with bilinear filtering
    (soft mask edges stay soft).
    """
    with Image.open(path) as im:
        if "A" in im.getbands():
            im = im.getchannel("A")
        elif im.mode != "L":
            im = im.convert("L")
        if size is not None and im.size != tuple(size):
            im = im.resize(tuple(size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def resize_masks(masks: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a [T, H, W] mask sequence to ``size`` = (width, height)."""
    if tuple(masks.shape[1:][::-1]) == tuple(size):
        return masks
    frames = [Image.fromarray((m * 255).astype(np.uint8)) for m in masks]
    frames = [f.resize(tuple(size), Image.BILINEAR) for f in frames]
    out = np.stack([np.asarray(f, dtype=np.float32) / 255.0 for f in frames])
    return out


#: sentinel for the EDT: larger than any possible squared distance
_EDT_LARGE = None  # set per-call from the mask shape

#: edge softness of the morph sigmoid, in pixels
_MORPH_SOFTNESS = 1.0

try:  # exact EDT at C speed; the pure-Python fallback is kept for CI/stubs
    from scipy.ndimage import distance_transform_edt as _scipy_edt
except ImportError:
    _scipy_edt = None


def _edt_1d_sq(f: np.ndarray) -> np.ndarray:
    """Exact 1D Euclidean distance transform (squared distances).

    Felzenszwalb-Huttenlocher lower-envelope of parabolas; O(n). ``f`` holds
    0.0 at foreground positions and a large sentinel elsewhere; the result is
    the squared distance to the nearest foreground element.
    """
    n = f.shape[0]
    v = np.zeros(n, dtype=np.int64)
    z = np.zeros(n + 1, dtype=np.float64)
    z[0] = -np.inf
    z[1] = np.inf
    k = 0
    for q in range(1, n):
        q2 = float(q * q)
        while True:
            vk = v[k]
            s = ((f[q] + q2) - (f[vk] + float(vk * vk))) / (2.0 * (q - vk))
            if s > z[k]:
                break
            k -= 1
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    d = np.empty(n, dtype=np.float64)
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        vk = v[k]
        d[q] = f[vk] + float(q - vk) * float(q - vk)
    return d


def _edt_2d(mask_binary: np.ndarray) -> np.ndarray:
    """Exact 2D Euclidean distance transform of a bool mask (True = foreground).

    Returns float64 [H, W] distances to the nearest foreground pixel. Uses
    scipy's C implementation when available (same exact distances, ~100x
    faster than the pure-Python Felzenszwalb-Huttenlocher fallback, which is
    kept for environments without scipy).
    """
    if _scipy_edt is not None:
        # scipy measures distance to the nearest ZERO of the input, so the
        # mask is inverted to get the distance to the nearest foreground
        return _scipy_edt(~mask_binary)
    h, w = mask_binary.shape
    large = float(h * h + w * w) + 1.0  # > any possible squared distance
    f = np.where(mask_binary, 0.0, large).astype(np.float64)
    for y in range(h):
        f[y, :] = _edt_1d_sq(f[y, :])
    for x in range(w):
        f[:, x] = np.sqrt(np.maximum(_edt_1d_sq(f[:, x]), 0.0))
    return f


def _signed_distance(mask_binary: np.ndarray) -> np.ndarray:
    """Signed distance field of a bool mask; positive inside, negative outside.

    Empty masks get a uniform -max(H, W)/2 field (so a shape morphs *in* from
    nothing); full masks get +max(H, W)/2 (so a shape morphs *out* to fill
    the frame).
    """
    h, w = mask_binary.shape
    inside = mask_binary.all()
    outside = not mask_binary.any()
    if outside:
        return np.full((h, w), -max(h, w) / 2.0, dtype=np.float64)
    if inside:
        return np.full((h, w), max(h, w) / 2.0, dtype=np.float64)
    d_fg = _edt_2d(mask_binary)  # distance to foreground (0 inside, >0 outside)
    d_bg = _edt_2d(~mask_binary)  # distance to background (0 outside, >0 inside)
    return d_bg - d_fg


def _sigmoid_stable(x: np.ndarray, softness: float) -> np.ndarray:
    """Sigmoid with clipped input so exp cannot overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(x / softness, -50.0, 50.0)))


def _shift(field: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a field by whole pixels; vacated pixels become 0."""
    h, w = field.shape
    out = np.zeros_like(field)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = field[src_y0:src_y1, src_x0:src_x1]
    return out


def _centroid(mask_binary: np.ndarray):
    """Centroid (y, x) of a bool mask, or None when it is empty."""
    ys, xs = np.where(mask_binary)
    if len(xs) == 0:
        return None
    return (float(ys.mean()), float(xs.mean()))


def interpolate_masks(
    keyframes: Dict[int, np.ndarray], count: int
) -> np.ndarray:
    """Morph a {frame_idx: [H, W] mask} dict to ``count`` frames (SDF level-set
    interpolation).

    Returns float32 [count, H, W] in [0, 1]. All keyframe masks must share
    one shape. ``count`` <= 0 raises ValueError. Exact keyframe frames return
    the original painted mask; frames outside the keyframe window (before the
    first / after the last keyframe) are all zeros (no mask).
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if not keyframes:
        raise ValueError("at least one keyframe is required")

    indices = sorted(keyframes)
    keys = {
        i: np.asarray(keyframes[i], dtype=np.float32) for i in indices
    }
    h, w = next(iter(keys.values())).shape
    out = np.zeros((count, h, w), dtype=np.float32)

    # keyframes at or beyond `count` are out of range; if none is in range,
    # no frame is a keyframe or between keyframes -> the mask stays empty
    in_range = [i for i in indices if i < count]
    if not in_range:
        return out

    # exact keyframe frames keep the original painted masks
    for i in in_range:
        out[i] = keys[i]

    if len(indices) == 1:
        return out

    sdfs = {i: _signed_distance(keys[i] >= 0.5) for i in indices}
    centroids = {i: _centroid(keys[i] >= 0.5) for i in indices}
    for lo, hi in zip(indices, indices[1:]):
        n_inner = hi - lo - 1
        if n_inner <= 0:
            continue
        # translation compensation: sample each field in the interpolated
        # frame so a translating shape slides instead of collapsing (plain
        # SDF blending vanishes when the translation exceeds the shape's
        # inscribed radius). Skip when either shape is empty (growth/shrink
        # morphs need no compensation).
        c_lo, c_hi = centroids[lo], centroids[hi]
        if c_lo is None or c_hi is None:
            dx, dy = 0.0, 0.0
        else:
            dx, dy = c_hi[1] - c_lo[1], c_hi[0] - c_lo[0]
        sdf_lo, sdf_hi = sdfs[lo], sdfs[hi]
        h, w = sdf_lo.shape
        out[lo + 1 : hi] = 0.0  # fill whole intermediate block per frame
        for t in range(lo + 1, hi):
            wf = (t - lo) / (hi - lo)
            # d(p) = (1-w)*sdf_lo(p - w*D) + w*sdf_hi(p + (1-w)*D)
            sx1 = int(np.floor(wf * dx + 0.5))
            sy1 = int(np.floor(wf * dy + 0.5))
            sx2 = int(np.floor((1.0 - wf) * dx + 0.5))
            sy2 = int(np.floor((1.0 - wf) * dy + 0.5))
            d = (1.0 - wf) * _shift(sdf_lo, sy1, sx1) + wf * _shift(sdf_hi, -sy2, -sx2)
            out[t] = _sigmoid_stable(d, _MORPH_SOFTNESS).astype(np.float32)
    return out


def parse_keyframes_widget(value: str) -> Dict[int, str]:
    """Parse the node's ``keyframes`` widget JSON into {frame_idx: filename}.

    Malformed input yields an empty dict (treated as "no keyframes"). Keys are
    coerced to ints.
    """
    if not value:
        return {}
    try:
        data = json.loads(value)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        try:
            if isinstance(v, str):
                out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out
