/**
 * Runs the frontend mask-morph math (web/lanpaint_mask_math.js) on the same
 * keyframes the Python backend uses and prints the per-frame masks, so a
 * pytest can pin the two implementations together.
 *
 * Usage: node parity_mask_math.mjs <input.json>
 *   input: {w, h, count, keyframes: {"<idx>": [alpha 0-255, ...]}}
 *   output (stdout): {w, h, count, data: [float ...]} per-frame flat arrays
 */

import { readFileSync } from "node:fs";

const data = JSON.parse(readFileSync(process.argv[2], "utf8"));

// the web dir has no package.json, so the module is loaded as ESM via a data: URL
const src = readFileSync(new URL("../web/lanpaint_mask_math.js", import.meta.url), "utf8");
const math = await import("data:text/javascript;base64," + Buffer.from(src).toString("base64"));

const { computeSDFEntry, MaskMorph } = math;
const { w, h, count, keyframes } = data;
const morph = new MaskMorph(w, h);
const indices = Object.keys(keyframes).map(Number).sort((a, b) => a - b);

const out = new Float32Array(count * w * h);
const sdfs = new Map();
for (const idx of indices) {
    const alpha = new Uint8ClampedArray(keyframes[idx]);
    out.set(alpha.map((a) => a / 255), idx * w * h); // exact keyframes keep the paint
    sdfs.set(idx, computeSDFEntry(alpha, w, h));
}
for (let p = 0; p < indices.length - 1; p++) {
    const lo = indices[p];
    const hi = indices[p + 1];
    const sdfLo = sdfs.get(lo);
    const sdfHi = sdfs.get(hi);
    for (let t = lo + 1; t < hi; t++) {
        const wf = (t - lo) / (hi - lo);
        out.set(morph.frame(sdfLo, sdfHi, wf), t * w * h);
    }
}
console.log(JSON.stringify({ w, h, count, data: Array.from(out) }));
