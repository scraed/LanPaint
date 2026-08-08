/**
 * Pure mask-morph math shared by the mask editor preview and the parity test
 * (tests/parity_mask_math.mjs). MUST mirror src/LanPaint/videomask.py —
 * interpolate_masks/_edt_2d/_signed_distance/_sigmoid_stable — so the preview
 * is pixel-equivalent to the backend output (the 0.5 level sets match
 * exactly: Math.round(v*255) >= 128 iff v >= 0.5).
 */

/** Exact 1D EDT (squared distances), Felzenszwalb-Huttenlocher, O(n). */
export function edt1DSq(f) {
    const n = f.length;
    const v = new Int32Array(n);
    const z = new Float64Array(n + 1);
    z[0] = -Infinity;
    z[1] = Infinity;
    let k = 0;
    for (let q = 1; q < n; q++) {
        const q2 = q * q;
        let s;
        for (;;) {
            const vk = v[k];
            s = (f[q] + q2 - (f[vk] + vk * vk)) / (2 * (q - vk));
            if (s > z[k]) break;
            k--;
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k + 1] = Infinity;
    }
    const d = new Float64Array(n);
    k = 0;
    for (let q = 0; q < n; q++) {
        while (z[k + 1] < q) k++;
        const vk = v[k];
        d[q] = f[vk] + (q - vk) * (q - vk);
    }
    return d;
}

/** Exact 2D EDT in place; f holds 0 at foreground, sentinel elsewhere. */
export function edt2D(f, w, h) {
    const row = new Float64Array(w);
    const col = new Float64Array(h);
    for (let y = 0; y < h; y++) {
        row.set(f.subarray(y * w, (y + 1) * w));
        f.set(edt1DSq(row), y * w);
    }
    for (let x = 0; x < w; x++) {
        for (let y = 0; y < h; y++) col[y] = f[y * w + x];
        const d = edt1DSq(col);
        for (let y = 0; y < h; y++) f[y * w + x] = Math.sqrt(Math.max(d[y], 0));
    }
    return f;
}

/**
 * Signed distance field of a mask's alpha channel (0-255, >= 128 = painted).
 * `alpha` must be a PURE alpha array of exactly w*h bytes (not RGBA).
 * Returns { field: Float64Array(w*h), cy, cx, hasShape } — positive inside.
 */
export function computeSDFEntry(alpha, w, h) {
    const n = w * h;
    if (alpha.length !== n) {
        throw new Error("computeSDFEntry expects a pure alpha array of w*h bytes (RGBA was passed?)");
    }
    const large = h * h + w * w + 1;
    const sentinel = Math.max(w, h) / 2;
    let any = false;
    let all = true;
    let sumY = 0;
    let sumX = 0;
    let count = 0;
    let f = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const on = alpha[i] >= 128;
        if (on) {
            any = true;
            sumY += (i / w) | 0;
            sumX += i % w;
            count++;
        } else {
            all = false;
        }
        f[i] = on ? 0.0 : large;
    }
    if (!any) {
        return { field: new Float64Array(n).fill(-sentinel), cy: 0, cx: 0, hasShape: false };
    }
    if (all) {
        // Python's np.where mean for an all-true mask is the frame center
        return {
            field: new Float64Array(n).fill(sentinel),
            cy: (h - 1) / 2,
            cx: (w - 1) / 2,
            hasShape: true
        };
    }
    const dFg = edt2D(f, w, h);
    f = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const on = alpha[i] >= 128;
        f[i] = on ? large : 0.0;
    }
    const dBg = edt2D(f, w, h);
    const field = new Float64Array(n);
    for (let i = 0; i < n; i++) field[i] = dBg[i] - dFg[i];
    return { field, cy: sumY / count, cx: sumX / count, hasShape: true };
}

/** Shift a field by whole pixels; vacated pixels become 0. */
export function shiftField(field, w, h, dy, dx) {
    const out = new Float64Array(w * h);
    const srcY0 = Math.max(0, -dy);
    const srcY1 = Math.min(h, h - dy);
    const srcX0 = Math.max(0, -dx);
    const srcX1 = Math.min(w, w - dx);
    for (let y = srcY0; y < srcY1; y++) {
        const srcRow = y * w;
        const dstRow = (y + dy) * w;
        for (let x = srcX0; x < srcX1; x++) {
            out[dstRow + x + dx] = field[srcRow + x];
        }
    }
    return out;
}

/** Edge softness of the morph sigmoid, in pixels (mirror _MORPH_SOFTNESS). */
export const MORPH_SOFTNESS = 1.0;

/** Exact mirror of videomask._sigmoid_stable: 1/(1+exp(-clip(d/soft, +-50))). */
export function sigmoidStable(d) {
    return 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, d / MORPH_SOFTNESS))));
}

/**
 * Per-frame SDF morph, mirroring interpolate_masks' inner loop exactly:
 *   d(p) = (1-w)*sdf_lo(p - w*D) + w*sdf_hi(p + (1-w)*D)
 *   mask = sigmoid(d)
 * Returns a Float32Array(w*h) in 0..1.
 */
export class MaskMorph {
    constructor(w, h) {
        this.w = w;
        this.h = h;
    }

    frame(loEntry, hiEntry, wf) {
        const w = this.w;
        const h = this.h;
        let sx1 = 0;
        let sy1 = 0;
        let sx2 = 0;
        let sy2 = 0;
        if (loEntry.hasShape && hiEntry.hasShape) {
            const dx = hiEntry.cx - loEntry.cx;
            const dy = hiEntry.cy - loEntry.cy;
            sx1 = Math.floor(wf * dx + 0.5);
            sy1 = Math.floor(wf * dy + 0.5);
            sx2 = Math.floor((1 - wf) * dx + 0.5);
            sy2 = Math.floor((1 - wf) * dy + 0.5);
        }
        const f1 = shiftField(loEntry.field, w, h, sy1, sx1);
        const f2 = shiftField(hiEntry.field, w, h, -sy2, -sx2);
        const out = new Float32Array(w * h);
        for (let i = 0; i < w * h; i++) {
            out[i] = sigmoidStable((1 - wf) * f1[i] + wf * f2[i]);
        }
        return out;
    }
}
