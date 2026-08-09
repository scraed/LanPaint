import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { computeSDFEntry, MaskMorph } from "./lanpaint_mask_math.js";

/**
 * LanPaint Video Mask Editor
 *
 * Opens a frame-scrubber + paint overlay on LanPaint_VideoMaskEditor nodes.
 * The user paints masks on keyframes; frames between keyframes show the
 * interpolated mask (same formula as src/LanPaint/videomask.py). Saving
 * uploads each keyframe mask as a PNG (mask in the alpha channel) and records
 * {"<frame>": "<file>.png"} in the node's hidden `keyframes` widget.
 */

const NODE_CLASS = "LanPaint_VideoMaskEditor";
const DEFAULT_FPS = 24;
const STRIP_H = 42; // waveform strip height in the node's live preview

let currentEditor = null;

/* ------------------------------------------------------------------ */
/* audio waveform (peaks + masked intervals)                           */
/* ------------------------------------------------------------------ */

const audioPeakCache = new Map(); // view url -> {mins, maxs, n, duration} | null

/**
 * Decode the audio track of a video (via the ComfyUI view endpoint) and
 * compute per-bucket min/max peaks. Returns null when there is no audio
 * track or decoding fails (the waveform simply is not shown).
 */
async function loadAudioPeaks(url) {
    if (audioPeakCache.has(url)) return audioPeakCache.get(url);
    const promise = (async () => {
        try {
            const resp = await fetch(url);
            if (!resp.ok) return null;
            const buf = await resp.arrayBuffer();
            const actx = new (window.AudioContext || window.webkitAudioContext)();
            let audio;
            try {
                audio = await actx.decodeAudioData(buf);
            } finally {
                actx.close();
            }
            const ch = audio.getChannelData(0);
            const N = 1200;
            const mins = new Float32Array(N);
            const maxs = new Float32Array(N);
            for (let i = 0; i < N; i++) {
                const s = Math.floor((i * ch.length) / N);
                const e = Math.max(s + 1, Math.floor(((i + 1) * ch.length) / N));
                let mn = 1;
                let mx = -1;
                for (let j = s; j < e; j++) {
                    const v = ch[j];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                mins[i] = mn;
                maxs[i] = mx;
            }
            return { mins, maxs, n: N, duration: audio.duration };
        } catch (_) {
            return null; // no audio track or undecodable: no waveform
        }
    })();
    audioPeakCache.set(url, promise);
    if (audioPeakCache.size > 8) {
        audioPeakCache.delete(audioPeakCache.keys().next().value);
    }
    return promise;
}

/** Parse the audio_mask widget JSON: [{start, end}, ...] in seconds. */
function parseAudioIntervals(value) {
    try {
        const data = JSON.parse(value || "[]");
        if (Array.isArray(data)) {
            return data
                .map((it) => ({
                    start: Number(it.start) || 0,
                    end: Number(it.end) || 0,
                }))
                .filter((it) => it.end > it.start)
                .sort((a, b) => a.start - b.start);
        }
    } catch (_) {
        /* fall through */
    }
    return [];
}

/** Full redraw of the waveform canvas: peaks, red intervals, playhead. */
function drawWaveform(canvas, peaks, duration, intervals, playhead) {
    const ctx = canvas.getContext("2d");
    const w = canvas.clientWidth || canvas.width || 200;
    const h = canvas.clientHeight || canvas.height || STRIP_H;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#17191d";
    ctx.fillRect(0, 0, w, h);

    // masked intervals (red, the same language as the video mask overlay)
    if (peaks && duration > 0) {
        for (const it of intervals) {
            const x0 = Math.max(0, (it.start / duration) * w);
            const x1 = Math.min(w, (it.end / duration) * w);
            ctx.fillStyle = "rgba(255, 40, 40, 0.30)";
            ctx.fillRect(x0, 0, x1 - x0, h);
            ctx.fillStyle = "#ff5252";
            ctx.fillRect(x0, 0, Math.max(1, x1 - x0), 2);
            ctx.fillRect(x0, h - 2, Math.max(1, x1 - x0), 2);
        }

        // min/max peak bars around the center line
        const mid = h / 2;
        const amp = (h / 2) * 0.92;
        ctx.strokeStyle = "#5f8ee8";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < peaks.n; i++) {
            const x = ((i + 0.5) / peaks.n) * w;
            const y0 = mid - Math.max(0.5, Math.abs(peaks.maxs[i]) * amp);
            const y1 = mid + Math.max(0.5, Math.abs(peaks.mins[i]) * amp);
            ctx.moveTo(x, y0);
            ctx.lineTo(x, y1);
        }
        ctx.stroke();
        // center line
        ctx.strokeStyle = "#3a4a63";
        ctx.beginPath();
        ctx.moveTo(0, mid);
        ctx.lineTo(w, mid);
        ctx.stroke();
    } else {
        ctx.fillStyle = "#3a3f47";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("no audio track", w / 2, h / 2 + 4);
    }

    // playhead marker
    if (playhead != null && peaks && duration > 0) {
        const x = (playhead / duration) * w;
        ctx.fillStyle = "rgba(255,255,255,0.9)";
        ctx.fillRect(x - 1, 0, 2, h);
    }
}

/**
 * Measure a video's fps with requestVideoFrameCallback (Chromium);
 * fallback DEFAULT_FPS. rVFC only fires when a frame is presented, so each
 * measurement is preceded by a seek, and every await has a timeout fallback.
 *
 * The measurement spans ~80% of the duration (not a short window): a 1-second
 * span has ±1 frame of uncertainty, so the editor and the node preview could
 * round to different integers (e.g. 29 vs 30) and their frame grids drifted
 * by a frame. Over the full duration the error is < 0.01 fps, so both
 * sessions round to the same value as the container's real rate.
 */
async function measureVideoFps(video) {
    if (typeof video.requestVideoFrameCallback !== "function") {
        return DEFAULT_FPS;
    }
    const meta = (timeoutMs) =>
        new Promise((resolve) => {
            let done = false;
            const finish = (mm) => {
                if (!done) {
                    done = true;
                    resolve(mm);
                }
            };
            video.requestVideoFrameCallback((_, mm) => finish(mm));
            setTimeout(() => finish(null), timeoutMs);
        });
    const seeked = (t) => {
        if (Math.abs(video.currentTime - t) < 1e-4) return Promise.resolve();
        video.currentTime = t;
        return new Promise((resolve) => {
            video.onseeked = resolve;
            setTimeout(resolve, 300);
        });
    };
    try {
        await seeked(0.001); // force a frame presentation from a paused state
        const snap1 = await meta(800);
        const dur = video.duration || 1;
        await seeked(Math.min(dur - 0.2, Math.max(0.5, dur * 0.8)));
        const snap2 = await meta(800);
        if (snap1 && snap2) {
            const dt = snap2.mediaTime - snap1.mediaTime;
            const df = snap2.presentedFrames - snap1.presentedFrames;
            if (dt > 0 && df > 0) {
                const fps = df / dt;
                if (fps > 1 && fps < 240) {
                    return Math.round(fps);
                }
            }
        }
    } catch (_) {
        /* keep default */
    }
    return DEFAULT_FPS;
}

/**
 * The keyframe mask timeline: owns the per-keyframe mask canvases and the
 * SDF morph preview. Shared by the editor overlay and the node's live
 * preview widget (web + backend use the same morphing math).
 */
class MaskTimeline {
    constructor() {
        this.width = 0;
        this.height = 0;
        this.keyframes = new Map(); // idx -> mask canvas (mask in alpha)
        this.sdfCache = new Map();  // idx -> computed SDF entry
        this.morph = null;          // MaskMorph (shared math with the backend)
    }

    setSize(width, height) {
        this.width = width;
        this.height = height;
        this.morph = new MaskMorph(width, height);
    }

    newMaskCanvas() {
        const c = document.createElement("canvas");
        c.width = this.width;
        c.height = this.height;
        return c;
    }

    setKeyframe(index, canvas) {
        this.keyframes.set(index, canvas);
        this.sdfCache.delete(index);
    }

    invalidate(index) {
        this.sdfCache.delete(index);
    }

    /** Make `index` a keyframe (copies the interpolated mask if needed). */
    ensureKeyframe(index) {
        if (this.keyframes.has(index)) return this.keyframes.get(index);
        const c = this.newMaskCanvas();
        const data = this.maskAt(index);
        if (data) c.getContext("2d").putImageData(data, 0, 0);
        this.keyframes.set(index, c);
        return c;
    }

    removeKeyframe(index) {
        this.keyframes.delete(index);
        this.sdfCache.delete(index);
    }

    /** SDF entry for a keyframe (cached; invalidated on paint/undo). */
    _sdfFor(index) {
        let entry = this.sdfCache.get(index);
        if (entry) return entry;
        const c = this.keyframes.get(index);
        const data = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
        // the shared math expects a pure alpha array (n bytes), not RGBA
        const n = c.width * c.height;
        const alpha = new Uint8ClampedArray(n);
        for (let i = 0; i < n; i++) alpha[i] = data[i * 4 + 3];
        entry = computeSDFEntry(alpha, c.width, c.height);
        this.sdfCache.set(index, entry);
        return entry;
    }

    /** Interpolated (morphed) mask at `index` as ImageData, or null. */
    maskAt(index) {
        const indices = [...this.keyframes.keys()].sort((a, b) => a - b);
        if (!indices.length) return null;

        // exact keyframe frames return the original painted mask
        if (this.keyframes.has(index)) {
            const c = this.keyframes.get(index);
            return c.getContext("2d").getImageData(0, 0, c.width, c.height);
        }
        // frames outside the keyframe window (before the first / after the
        // last keyframe) have no mask at all
        if (index < indices[0] || index > indices[indices.length - 1]) {
            return null;
        }
        const lo = indices.filter((i) => i <= index).pop() ?? indices[0];
        const hi = indices.find((i) => i >= index) ?? indices[indices.length - 1];

        // SDF morph between the neighboring keyframes, with translation
        // compensation so a moving shape slides instead of collapsing. The
        // math lives in lanpaint_mask_math.js and mirrors the backend's
        // interpolate_masks exactly (same sigmoid, same shifts).
        const sdfLo = this._sdfFor(lo);
        const sdfHi = this._sdfFor(hi);
        const w = this.width;
        const h = this.height;
        const wf = (index - lo) / (hi - lo);
        const vals = this.morph.frame(sdfLo, sdfHi, wf);
        const out = new ImageData(w, h);
        const n = w * h;
        for (let i = 0; i < n; i++) {
            const k = i * 4;
            out.data[k] = 255;
            out.data[k + 1] = 255;
            out.data[k + 2] = 255;
            out.data[k + 3] = Math.round(vals[i] * 255);
        }
        return out;
    }
}

/* ------------------------------------------------------------------ */
/* editor                                                              */
/* ------------------------------------------------------------------ */

class VideoMaskEditor {
    constructor(node) {
        this.node = node;
        this.video = null;          // HTMLVideoElement
        this.fps = DEFAULT_FPS;
        this.frameCount = 0;
        this.videoW = 0;
        this.videoH = 0;
        this.current = 0;
        this.timeline = new MaskTimeline();
        this.undo = new Map();      // idx -> [ImageData]
        this.frameCache = new Map(); // idx -> canvas (LRU)
        this.brush = { size: 40, hardness: 0.6, erase: false, spacePan: false };
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.el = null;
        this.canvas = null;         // display canvas (stage-sized)
        this.frameCanvas = null;    // current video frame
        this.maskCanvas = null;     // current editing mask (alpha = mask)
        this.overlayCanvas = null;  // red tint composited from maskCanvas
        this.seekPending = false;
        this.audioIntervals = []; // [{start, end}] seconds, regenerated audio
        this.peaks = null;         // decoded waveform peaks
        this.waveCanvas = null;
        this.waveDrag = null;      // {start, end} while dragging a new interval
        this._maskVideo = null;    // last video for which masks were auto-loaded
    }

    /* ---------------- lifecycle ---------------- */

    async open() {
        const filename = this.getVideoFilename();
        if (!filename) {
            alert(
                "Pick a video file in the node first (the 'video' dropdown), then open the editor."
            );
            return;
        }
        const div = document.createElement("div");
        div.id = "lanpaint-video-mask-editor";
        div.style.cssText =
            "position:fixed;inset:0;z-index:9999;background:#1a1a1a;color:#ddd;" +
            "display:flex;flex-direction:column;font-family:sans-serif;user-select:none;";
        div.innerHTML = this._layoutHTML();
        document.body.appendChild(div);
        this.el = div;
        this._bindEvents();

        try {
            await this._loadVideo(filename);
            if (!this._metaLoaded) {
                await this._loadSavedKeyframes();
            }
            this.peaks = await loadAudioPeaks(this.video.currentSrc);
            this._loadAudioIntervals();
            await this.showFrame(0);
            this._drawWave();
            this._fit();
        } catch (err) {
            console.error("[LanPaint VideoMaskEditor] open failed:", err);
            alert("Failed to open the video: " + err.message);
            this.close();
        }
    }

    close() {
        if (this.el) {
            window.removeEventListener("keydown", this._onKeyDown);
            window.removeEventListener("keyup", this._onKeyUp);
            this.el.remove();
            this.el = null;
        }
        currentEditor = null;
    }

    getVideoFilename() {
        const w = this.node.widgets?.find((x) => x.name === "video");
        return w && typeof w.value === "string" ? w.value : null;
    }

    getKeyframesWidget() {
        return this.node.widgets?.find((x) => x.name === "keyframes");
    }

    getAudioMaskWidget() {
        return this.node.widgets?.find((x) => x.name === "audio_mask");
    }

    _loadAudioIntervals() {
        this.audioIntervals = parseAudioIntervals(this.getAudioMaskWidget()?.value);
    }

    /** Time of the current frame (the playhead position on the waveform). */
    _playheadTime() {
        return this.fps > 0 ? this.current / this.fps : 0;
    }

    _audioDuration() {
        return this.peaks?.duration || this.video?.duration || 0;
    }

    _drawWave() {
        if (!this.el || !this.waveCanvas) return;
        const intervals = this.waveDrag
            ? [...this.audioIntervals, this.waveDrag]
            : this.audioIntervals;
        drawWaveform(
            this.waveCanvas,
            this.peaks,
            this._audioDuration(),
            intervals,
            this._playheadTime()
        );
    }

    /* ---------------- audio interval painting ---------------- */

    _bindWaveEvents() {
        this.waveCanvas = this.el.querySelector("#lpvme-wave");
        const duration = () => this._audioDuration();

        this.waveCanvas.addEventListener("pointerdown", (e) => {
            if (!this.peaks || duration() <= 0) return;
            e.preventDefault();
            const rect = this.waveCanvas.getBoundingClientRect();
            const t = ((e.clientX - rect.left) / rect.width) * duration();
            // click inside an existing interval removes it
            const hit = this.audioIntervals.find((it) => t >= it.start && t <= it.end);
            if (hit) {
                this.audioIntervals = this.audioIntervals.filter((x) => x !== hit);
                this.waveDrag = null;
                this._drawWave();
                return;
            }
            this.waveDrag = { start: t, end: t };
            this.waveCanvas.setPointerCapture(e.pointerId);
            this._drawWave();
        });
        this.waveCanvas.addEventListener("pointermove", (e) => {
            if (!this.waveDrag) return;
            const rect = this.waveCanvas.getBoundingClientRect();
            const t = ((e.clientX - rect.left) / rect.width) * duration();
            this.waveDrag.end = Math.min(duration(), Math.max(0, t));
            this._drawWave();
        });
        const finishWaveDrag = () => {
            if (!this.waveDrag) return;
            const d = this.waveDrag;
            this.waveDrag = null;
            // a click without a drag creates nothing
            if (d.end - d.start > 0.05) {
                this.audioIntervals.push({
                    start: Math.min(d.start, d.end),
                    end: Math.max(d.start, d.end),
                });
                this.audioIntervals.sort((a, b) => a.start - b.start);
            }
            this._drawWave();
        };
        this.waveCanvas.addEventListener("pointerup", finishWaveDrag);
        this.waveCanvas.addEventListener("pointercancel", finishWaveDrag);
    }

    /* ---------------- video ---------------- */

    async _loadVideo(filename) {
        const url =
            api.apiURL(
                "/view?filename=" + encodeURIComponent(filename) + "&type=input"
            ) + app.getRandParam();
        const video = document.createElement("video");
        video.muted = true;
        video.playsInline = true;
        video.preload = "auto";
        video.src = url;
        await new Promise((resolve, reject) => {
            video.onloadedmetadata = resolve;
            video.onerror = () => reject(new Error("cannot load video file: " + filename));
        });
        this.video = video;
        // dimensions can arrive later than metadata (some containers)
        for (let tries = 0; tries < 50 && (!video.videoWidth || !video.videoHeight); tries++) {
            await new Promise((resolve) => {
                video.onseeked = resolve;
                setTimeout(resolve, 120);
            });
        }
        this.videoW = video.videoWidth;
        this.videoH = video.videoHeight;
        this.timeline.setSize(this.videoW, this.videoH);
        this.overlayCanvas = this.timeline.newMaskCanvas();
        // render the first frame (muted play is allowed without a gesture)
        try {
            await video.play();
        } catch (_) {
            /* ignored */
        }
        video.pause();
        this.fps = await measureVideoFps(video);
        this.frameCount = Math.max(1, Math.round(video.duration * this.fps));
        this._metaLoaded = await this._autoLoadMaskMeta(filename);
    }

    /**
     * Fetch `/lanpaint/video_mask_meta` for the video and, if metadata is
     * present, upload the base64 keyframes as PNGs and populate the timeline
     * and widget. Returns true when metadata was found and loaded.
     */
    async _autoLoadMaskMeta(filename) {
        try {
            const url =
                api.apiURL(
                    "/lanpaint/video_mask_meta?filename=" + encodeURIComponent(filename)
                ) + "&rand=" + Math.random();
            const resp = await fetch(url);
            if (!resp.ok) throw new Error("HTTP " + resp.status);
            const json = await resp.json();
            if (!json.found || !json.payload) {
                // the video carries no mask metadata: the workflow's masks are
                // meaningless for it, so clear them (the video is the source
                // of truth for masks)
                this.timeline.keyframes.clear();
                this.timeline.sdfCache.clear();
                this.undo.clear();
                this.audioIntervals = [];
                const kw = this.getKeyframesWidget();
                if (kw) kw.value = "{}";
                const aw = this.getAudioMaskWidget();
                if (aw) aw.value = "[]";
                return false;
            }
            const payload = json.payload;
            // clear existing keyframes (the metadata is the source of truth)
            this.timeline.keyframes.clear();
            this.timeline.sdfCache.clear();
            this.undo.clear();

            const keyframeData = payload.keyframes || {};
            const filenames = {};
            const stamp = Date.now();
            for (const [idxStr, b64png] of Object.entries(keyframeData)) {
                const idx = parseInt(idxStr, 10);
                if (isNaN(idx) || idx < 0 || idx >= this.frameCount) continue;
                try {
                    const img = await new Promise((resolve, reject) => {
                        const im = new Image();
                        im.onload = () => resolve(im);
                        im.onerror = () => reject(new Error("decode failed"));
                        im.src = "data:image/png;base64," + b64png;
                    });
                    const c = this.timeline.newMaskCanvas();
                    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
                    this.timeline.setKeyframe(idx, c);

                    // upload so the widget stores a server filename
                    const blob = await new Promise((r) => c.toBlob(r, "image/png"));
                    if (!blob) continue;
                    const name = "lanpaint_kf_" + stamp + "_" + idx + ".png";
                    const fd = new FormData();
                    fd.append("image", blob, name);
                    fd.append("type", "input");
                    const upResp = await api.fetchApi("/upload/image", {
                        method: "POST",
                        body: fd,
                    });
                    if (!upResp.ok) continue;
                    const upData = await upResp.json();
                    filenames[idx] = upData.name;
                } catch (e) {
                    console.warn("[LanPaint] auto-load keyframe decode failed:", e);
                }
            }

            const kw = this.getKeyframesWidget();
            if (kw) kw.value = JSON.stringify(filenames);
            const aw = this.getAudioMaskWidget();
            if (aw) aw.value = JSON.stringify(payload.audio_intervals || "[]");
            this._loadAudioIntervals();
            this._maskVideo = filename;

            // keep widget_values in sync
            if (this.node.widgets_values && this.node.widgets) {
                for (const ww of [kw, aw]) {
                    if (!ww) continue;
                    const wi = this.node.widgets.indexOf(ww);
                    if (wi >= 0) this.node.widgets_values[wi] = ww.value;
                }
            }

            console.log(
                "[LanPaint VideoMaskEditor] auto-loaded",
                Object.keys(filenames).length,
                "keyframes from mp4 metadata"
            );
            return true;
        } catch (err) {
            console.warn("[LanPaint] video_mask_meta fetch failed:", err);
            // no metadata could be read: treat like not-found and clear
            this.timeline.keyframes.clear();
            this.timeline.sdfCache.clear();
            this.undo.clear();
            this.audioIntervals = [];
            const kw = this.getKeyframesWidget();
            if (kw) kw.value = "{}";
            const aw = this.getAudioMaskWidget();
            if (aw) aw.value = "[]";
            return false;
        }
    }

    /** Canvas for frame `index` (cached, LRU). */
    async frameAt(index) {
        const cached = this.frameCache.get(index);
        if (cached) return cached;
        // +1e-6 nudge: index/fps can round just below the frame boundary in
        // float, and the browser presents the frame CONTAINING the time —
        // without the nudge the seek can land on frame index-1.
        const target = Math.min(
            index / this.fps + 1e-6,
            Math.max(0, (this.video.duration || 0) - 1e-3)
        );
        if (Math.abs(this.video.currentTime - target) >= 1e-4) {
            this.video.currentTime = target;
            await new Promise((resolve) => {
                this.video.onseeked = resolve;
                setTimeout(resolve, 400); // safety: some videos seek lazily
            });
        }
        const c = document.createElement("canvas");
        c.width = this.videoW;
        c.height = this.videoH;
        c.getContext("2d").drawImage(this.video, 0, 0, this.videoW, this.videoH);
        if (this.frameCache.size >= 32) {
            this.frameCache.delete(this.frameCache.keys().next().value);
        }
        this.frameCache.set(index, c);
        return c;
    }

    /* ---------------- masks (delegated to this.timeline) ---------------- */

    /** Make `index` a keyframe (copies the interpolated mask if needed). */
    ensureKeyframe(index) {
        const kf = this.timeline.ensureKeyframe(index);
        this._refreshSlider();
        return kf;
    }

    /* ---------------- drawing ---------------- */

    _draw() {
        if (!this.el || !this.frameCanvas) return;
        const ctx = this.canvas.getContext("2d");
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);
        ctx.drawImage(this.frameCanvas, 0, 0);
        if (this.maskCanvas) {
            const oc = this.overlayCanvas;
            const octx = oc.getContext("2d");
            octx.clearRect(0, 0, oc.width, oc.height);
            octx.fillStyle = "rgba(255, 40, 40, 0.75)";
            octx.fillRect(0, 0, oc.width, oc.height);
            octx.globalCompositeOperation = "destination-in";
            octx.drawImage(this.maskCanvas, 0, 0);
            octx.globalCompositeOperation = "source-over";
            ctx.drawImage(oc, 0, 0);
        }
        ctx.restore();
    }

    async showFrame(index) {
        this.current = index;
        this.frameCanvas = await this.frameAt(index);
        this.maskCanvas = this.timeline.newMaskCanvas();
        const data = this.timeline.maskAt(index);
        if (data) this.maskCanvas.getContext("2d").putImageData(data, 0, 0);
        this._syncUI();
        this._draw();
    }

    /** Drain pending slider jumps to the latest frame. */
    async _showDebounced() {
        if (this.seekPending) return;
        this.seekPending = true;
        try {
            for (;;) {
                const target = this.current;
                await this.showFrame(target);
                if (this.current === target) break;
            }
        } finally {
            this.seekPending = false;
        }
    }

    _stroke(clientX, clientY) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (clientX - rect.left - this.panX) / this.zoom;
        const y = (clientY - rect.top - this.panY) / this.zoom;
        const ctx = this.maskCanvas.getContext("2d");
        const r = Math.max(1, this.brush.size / 2);
        const hard = Math.min(1, Math.max(0, this.brush.hardness));
        const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
        grad.addColorStop(0, "rgba(255,255,255,1)");
        grad.addColorStop(hard, "rgba(255,255,255,1)");
        grad.addColorStop(1, "rgba(255,255,255,0)");
        ctx.fillStyle = grad;
        ctx.globalCompositeOperation = this.brush.erase ? "destination-out" : "source-over";
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        this.timeline.invalidate(this.current); // the mask changed: SDF is stale
        this._draw();
    }

    /* ---------------- keyframe ops ---------------- */

    pushUndo(index) {
        const c = this.timeline.keyframes.get(index);
        if (!c) return;
        const stack = this.undo.get(index) || [];
        stack.push(c.getContext("2d").getImageData(0, 0, c.width, c.height));
        if (stack.length > 20) stack.shift();
        this.undo.set(index, stack);
    }

    undoCurrent() {
        const stack = this.undo.get(this.current);
        if (stack?.length && this.timeline.keyframes.has(this.current)) {
            this.timeline.keyframes
                .get(this.current)
                .getContext("2d")
                .putImageData(stack.pop(), 0, 0);
            this.timeline.invalidate(this.current);
            this._draw();
        }
    }

    removeKeyframe(index) {
        this.timeline.removeKeyframe(index);
        this.undo.delete(index);
        this.showFrame(index);
    }

    /* ---------------- save / load ---------------- */

    async _loadSavedKeyframes() {
        const w = this.getKeyframesWidget();
        if (!w) return;
        let data = {};
        try {
            data = JSON.parse(w.value || "{}");
        } catch (_) {
            return;
        }
        for (const [idxStr, filename] of Object.entries(data)) {
            const idx = parseInt(idxStr, 10);
            if (isNaN(idx) || idx < 0 || idx >= this.frameCount) continue;
            try {
                const url =
                    api.apiURL(
                        "/view?filename=" + encodeURIComponent(filename) + "&type=input"
                    ) + app.getRandParam();
                const img = await new Promise((resolve, reject) => {
                    const im = new Image();
                    im.crossOrigin = "anonymous";
                    im.onload = () => resolve(im);
                    im.onerror = () => reject(new Error("cannot load " + filename));
                    im.src = url;
                });
                const c = this.timeline.newMaskCanvas();
                c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
                this.timeline.setKeyframe(idx, c); // PNG stores the mask in alpha
            } catch (err) {
                console.warn("[LanPaint VideoMaskEditor] keyframe load failed:", err);
            }
        }
    }

    async _exportMaskVideo() {
        const filename = this.getVideoFilename();
        if (!filename) {
            alert("No video file selected.");
            return;
        }
        if (this.timeline.keyframes.size === 0 && this.audioIntervals.length === 0) {
            alert("No keyframes or audio intervals to export.");
            return;
        }
        try {
            const keyframes = {};
            for (const [idx, canvas] of this.timeline.keyframes) {
                keyframes[idx] = canvas
                    .toDataURL("image/png")
                    .replace(/^data:image\/png;base64,/, "");
            }
            const body = {
                filename: filename,
                keyframes: keyframes,
                audio_intervals: this.audioIntervals,
                fps: this.fps,
            };
            const resp = await api.fetchApi("/lanpaint/export_mask_video", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!resp.ok) {
                const text = await resp.text().catch(() => "");
                throw new Error("Export failed (" + resp.status + "): " + text);
            }
            const result = await resp.json();
            this._maskVideo = filename;
            // offer a native save dialog (choose location + name) when available;
            // the input/ copy from the export stays as the ComfyUI-visible copy
            if (typeof window.showSaveFilePicker === "function") {
                try {
                    const viewUrl =
                        api.apiURL(
                            "/view?filename=" + encodeURIComponent(result.filename) + "&type=input"
                        ) + app.getRandParam();
                    const buf = await (await fetch(viewUrl)).arrayBuffer();
                    const handle = await window.showSaveFilePicker({
                        suggestedName: result.filename,
                        types: [
                            {
                                description: "MP4 video",
                                accept: { "video/mp4": [".mp4"] },
                            },
                        ],
                    });
                    const writable = await handle.createWritable();
                    await writable.write(new Uint8Array(buf));
                    await writable.close();
                    this._flash(
                        "Saved to " + handle.name + " (also in input: " + result.filename + ")"
                    );
                } catch (pickErr) {
                    // user cancelled the picker or the API failed: the input/
                    // copy is still there and usable
                    this._flash("Exported to input: " + (result.path || result.filename));
                }
            } else {
                this._flash("Exported: " + (result.path || result.filename));
            }

            // If the video widget points at the source file, refresh it to the
            // newly exported masked file.
            const vw = this.node.widgets?.find((x) => x.name === "video");
            if (vw && vw.value === filename && vw.callback) {
                vw.value = result.filename;
                if (this.node.widgets_values) {
                    const vi = this.node.widgets.indexOf(vw);
                    if (vi >= 0) this.node.widgets_values[vi] = result.filename;
                }
                try {
                    vw.callback(result.filename);
                } catch (_) {
                    /* ignore callback errors */
                }
            }
        } catch (err) {
            alert("Export failed: " + err.message);
        }
    }

    async save() {
        const w = this.getKeyframesWidget();
        if (!w) return;
        const filenames = {};
        const stamp = Date.now();
        for (const [idx, canvas] of this.timeline.keyframes) {
            const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
            if (!blob) continue;
            const name = "lanpaint_kf_" + stamp + "_" + idx + ".png";
            const fd = new FormData();
            fd.append("image", blob, name);
            fd.append("type", "input");
            const resp = await api.fetchApi("/upload/image", { method: "POST", body: fd });
            if (!resp.ok) throw new Error("upload failed (" + resp.status + ")");
            const data = await resp.json();
            filenames[idx] = data.name;
        }
        w.value = JSON.stringify(filenames);
        const aw = this.getAudioMaskWidget();
        if (aw) {
            aw.value = JSON.stringify(this.audioIntervals);
        }
        if (this.node.widgets_values && this.node.widgets) {
            for (const ww of [w, aw]) {
                const wi = this.node.widgets.indexOf(ww);
                if (wi >= 0) this.node.widgets_values[wi] = ww.value;
            }
        }
        app.canvas.setDirty(true);
        console.log(
            "[LanPaint VideoMaskEditor] saved",
            Object.keys(filenames).length,
            "keyframes,",
            this.audioIntervals.length,
            "audio intervals"
        );
    }

    /* ---------------- UI ---------------- */

    _layoutHTML() {
        return `
        <style>
          #lanpaint-video-mask-editor button {
            background:#3a3a3a;color:#eee;border:1px solid #555;border-radius:4px;
            padding:4px 12px;cursor:pointer;font-size:13px;
          }
          #lanpaint-video-mask-editor button:hover { background:#4a4a4a; }
          #lanpaint-video-mask-editor button.lpvme-active { background:#ff5252; border-color:#ff5252; }
        </style>
        <div style="display:flex;align-items:center;gap:12px;padding:8px 14px;background:#252525;border-bottom:1px solid #444;">
          <strong style="color:#fff;">Video Mask Editor</strong>
          <span id="lpvme-frame-label">frame 0 / 0</span>
          <span style="flex:1"></span>
          <button id="lpvme-save" title="Upload keyframes and write them into the node">Save</button>
          <button id="lpvme-export" title="Remux the source video with mask metadata">导出 mask 视频</button>
          <button id="lpvme-close" title="Close (Escape)">Close</button>
        </div>
        <div id="lpvme-stage" style="flex:1;position:relative;overflow:hidden;background:#0f0f0f;">
          <canvas id="lpvme-canvas" style="position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;"></canvas>
          <div style="position:absolute;left:10px;bottom:10px;background:#000a;color:#eee;padding:4px 8px;border-radius:4px;font-size:12px;">
            draw: left drag &nbsp;|&nbsp; erase: E &nbsp;|&nbsp; wheel: zoom &nbsp;|&nbsp; middle or Space+drag: pan &nbsp;|&nbsp; undo: Ctrl+Z
          </div>
        </div>
        <div style="height:64px;background:#1f2226;border-top:1px solid #444;position:relative;">
          <canvas id="lpvme-wave" style="position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;"></canvas>
          <div style="position:absolute;right:10px;top:4px;background:#000a;color:#eee;padding:2px 8px;border-radius:4px;font-size:11px;">
            audio mask: drag to mark (red = regenerate) &nbsp;|&nbsp; click a red region to clear
          </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:8px;padding:10px 14px;background:#252525;border-top:1px solid #444;">
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="position:relative;flex:1;">
              <input id="lpvme-slider" type="range" min="0" max="1" value="0" style="width:100%;display:block;">
              <div id="lpvme-dots" style="position:absolute;inset:0;pointer-events:none;"></div>
            </div>
            <button id="lpvme-add-kf" title="Make the current frame a keyframe">Add Keyframe</button>
            <button id="lpvme-del-kf" title="Remove the keyframe at the current frame">Remove Keyframe</button>
            <button id="lpvme-undo" title="Undo last stroke (Ctrl+Z)">Undo</button>
          </div>
          <div style="display:flex;align-items:center;gap:10px;font-size:12px;">
            <button id="lpvme-brush" class="lpvme-active">Brush</button>
            <button id="lpvme-eraser">Eraser</button>
            <label>Size <input id="lpvme-size" type="range" min="2" max="200" value="40" style="width:120px;"></label>
            <label>Hardness <input id="lpvme-hard" type="range" min="0" max="100" value="60" style="width:120px;"></label>
            <span style="flex:1"></span>
            <button id="lpvme-fit" title="Fit frame to stage">Fit</button>
            <button id="lpvme-zoom-in" title="Zoom in">+</button>
            <button id="lpvme-zoom-out" title="Zoom out">-</button>
          </div>
        </div>`;
    }

    _syncUI() {
        if (!this.el) return;
        const label = this.el.querySelector("#lpvme-frame-label");
        label.textContent = "frame " + this.current + " / " + (this.frameCount - 1);
        const slider = this.el.querySelector("#lpvme-slider");
        slider.max = String(Math.max(1, this.frameCount - 1));
        slider.value = String(this.current);
        this._refreshSlider();
    }

    _refreshSlider() {
        const dots = this.el?.querySelector("#lpvme-dots");
        if (!dots) return;
        dots.innerHTML = "";
        const max = Math.max(1, this.frameCount - 1);
        for (const idx of this.timeline.keyframes.keys()) {
            const d = document.createElement("div");
            d.style.cssText =
                "position:absolute;top:2px;width:8px;height:8px;border-radius:50%;background:#ff5252;" +
                "transform:translateX(-50%);cursor:pointer;pointer-events:auto;";
            d.style.left = (idx / max) * 100 + "%";
            d.title = "keyframe " + idx + " (click: jump, right-click: remove)";
            d.addEventListener("click", (e) => {
                e.stopPropagation();
                this.showFrame(idx);
            });
            d.addEventListener("contextmenu", (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.removeKeyframe(idx);
            });
            dots.appendChild(d);
        }
    }

    _fit() {
        if (!this.el) return; // may be called from a late open() after close()
        const stage = this.el.querySelector("#lpvme-stage");
        const w = stage.clientWidth;
        const h = stage.clientHeight;
        if (!w || !h) return;
        this.zoom = Math.min(w / this.videoW, h / this.videoH) * 0.95;
        this.panX = (w - this.videoW * this.zoom) / 2;
        this.panY = (h - this.videoH * this.zoom) / 2;
        this._draw();
    }

    _bindEvents() {
        this.canvas = this.el.querySelector("#lpvme-canvas");
        this._bindWaveEvents();

        const stage = this.el.querySelector("#lpvme-stage");
        const resize = () => {
            this.canvas.width = stage.clientWidth;
            this.canvas.height = stage.clientHeight;
            this._draw();
        };
        resize();
        new ResizeObserver(resize).observe(stage);

        // slider (debounced frame jump that drains to the latest position)
        this.el.querySelector("#lpvme-slider").addEventListener("input", (e) => {
            this.current = parseInt(e.target.value, 10);
            const label = this.el.querySelector("#lpvme-frame-label");
            label.textContent = "frame " + this.current + " / " + (this.frameCount - 1);
            this._drawWave();
            this._showDebounced();
        });

        this.el.querySelector("#lpvme-save").addEventListener("click", async () => {
            try {
                await this.save();
                this._flash("Saved " + this.timeline.keyframes.size + " keyframes");
            } catch (err) {
                alert("Save failed: " + err.message);
            }
        });
        this.el.querySelector("#lpvme-export").addEventListener("click", () => {
            this._exportMaskVideo();
        });
        this.el.querySelector("#lpvme-close").addEventListener("click", () => this.close());
        this.el.querySelector("#lpvme-add-kf").addEventListener("click", () => {
            this.ensureKeyframe(this.current);
            this.showFrame(this.current);
        });
        this.el.querySelector("#lpvme-del-kf").addEventListener("click", () => {
            this.removeKeyframe(this.current);
        });
        this.el.querySelector("#lpvme-undo").addEventListener("click", () => this.undoCurrent());

        const brushBtn = this.el.querySelector("#lpvme-brush");
        const eraserBtn = this.el.querySelector("#lpvme-eraser");
        brushBtn.addEventListener("click", () => {
            this.brush.erase = false;
            brushBtn.classList.add("lpvme-active");
            eraserBtn.classList.remove("lpvme-active");
        });
        eraserBtn.addEventListener("click", () => {
            this.brush.erase = true;
            eraserBtn.classList.add("lpvme-active");
            brushBtn.classList.remove("lpvme-active");
        });
        this.el.querySelector("#lpvme-size").addEventListener("input", (e) => {
            this.brush.size = parseInt(e.target.value, 10);
        });
        this.el.querySelector("#lpvme-hard").addEventListener("input", (e) => {
            this.brush.hardness = parseInt(e.target.value, 10) / 100;
        });
        this.el.querySelector("#lpvme-fit").addEventListener("click", () => this._fit());
        this.el.querySelector("#lpvme-zoom-in").addEventListener("click", () => {
            this.zoom *= 1.25;
            this._draw();
        });
        this.el.querySelector("#lpvme-zoom-out").addEventListener("click", () => {
            this.zoom /= 1.25;
            this._draw();
        });

        // pan helper (middle-drag or space+drag)
        const startPan = (px, py) => {
            const start = { x: this.panX, y: this.panY, px, py };
            const onMove = (ev) => {
                this.panX = start.x + (ev.clientX - start.px);
                this.panY = start.y + (ev.clientY - start.py);
                this._draw();
            };
            const onUp = () => {
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
            };
            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
        };

        // painting
        let painting = false;
        this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
        this.canvas.addEventListener("pointerdown", (e) => {
            if (e.button === 1) {
                e.preventDefault();
                startPan(e.clientX, e.clientY);
                return;
            }
            if (e.button !== 0) return;
            if (this.brush.spacePan) {
                startPan(e.clientX, e.clientY);
                return;
            }
            e.preventDefault();
            const kf = this.ensureKeyframe(this.current);
            this.maskCanvas = kf; // paint directly on the keyframe canvas
            this.pushUndo(this.current);
            painting = true;
            this.canvas.setPointerCapture(e.pointerId);
            this._stroke(e.clientX, e.clientY);
        });
        this.canvas.addEventListener("pointermove", (e) => {
            if (painting) this._stroke(e.clientX, e.clientY);
        });
        const stopPainting = () => {
            painting = false;
        };
        this.canvas.addEventListener("pointerup", stopPainting);
        this.canvas.addEventListener("pointercancel", stopPainting);

        // wheel zoom at cursor
        this.canvas.addEventListener(
            "wheel",
            (e) => {
                e.preventDefault();
                const rect = this.canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
                const worldX = (mx - this.panX) / this.zoom;
                const worldY = (my - this.panY) / this.zoom;
                this.zoom *= factor;
                this.panX = mx - worldX * this.zoom;
                this.panY = my - worldY * this.zoom;
                this._draw();
            },
            { passive: false }
        );

        window.addEventListener("keydown", this._onKeyDown);
        window.addEventListener("keyup", this._onKeyUp);
    }

    _onKeyDown = (e) => {
        if (e.key === "Escape") {
            this.close();
            return;
        }
        if (e.key === " ") {
            this.brush.spacePan = true;
            e.preventDefault();
        }
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
            e.preventDefault();
            this.undoCurrent();
        }
        if (e.key.toLowerCase() === "e" && !e.ctrlKey && !e.metaKey) {
            this.brush.erase = !this.brush.erase;
            this.el?.querySelector("#lpvme-brush")?.classList.toggle("lpvme-active", !this.brush.erase);
            this.el?.querySelector("#lpvme-eraser")?.classList.toggle("lpvme-active", this.brush.erase);
        }
        e.stopPropagation();
    };

    _onKeyUp = (e) => {
        if (e.key === " ") this.brush.spacePan = false;
    };

    _flash(message) {
        const stage = this.el?.querySelector("#lpvme-stage");
        if (!stage) return; // the editor may have been closed mid-save
        const div = document.createElement("div");
        div.textContent = message;
        div.style.cssText =
            "position:absolute;top:12px;left:50%;transform:translateX(-50%);" +
            "background:#2e7d32;color:#fff;padding:6px 14px;border-radius:4px;z-index:5;";
        stage.appendChild(div);
        setTimeout(() => div.remove(), 1800);
    }
}

/* ------------------------------------------------------------------ */
/* node live preview (video + mask, like a video loading node)         */
/* ------------------------------------------------------------------ */

class NodeMaskPreview {
    constructor(node) {
        this.node = node;
        node.previewMediaType = "video"; // routes the built-in preview to the video path
        this.timeline = null;
        this.fps = DEFAULT_FPS;
        this.frameCount = 0;
        this.overlayCanvas = null;
        this.raf = 0;
        this.disposed = false;
        this.lastKeyframesValue = null;
        this.lastMaskIdx = null;
        this.lastVideoEl = null;
        this.lastSrc = null;
        this.stripCanvas = null;
        this.audioPeaks = null;
        this.lastAudioSrc = null;
        this.lastAudioMaskValue = null;
        this.lastDrawnValue = null;
        this.lastPlayheadX = null;
        this._sizeWrapped = false;
        this._maskVideo = null;    // last video for which masks were auto-loaded
        this._tick();
    }

    _getVideoFilename() {
        const w = this.node.widgets?.find((x) => x.name === "video");
        return w && typeof w.value === "string" ? w.value : null;
    }

    _keyframesWidget() {
        return this.node.widgets?.find((w) => w.name === "keyframes");
    }

    /**
     * The frontend attaches its OWN LoadVideo-style preview to this node
     * (it has a VIDEO output), so there is exactly ONE player: the built-in
     * `video-preview` widget. We only inject a transparent mask overlay into
     * its container and keep the per-frame mask in sync with playback.
     */
    _ensureOverlay() {
        const container = this.node.videoContainer;
        if (!container) return null;
        container.style.position = "relative"; // for the absolute overlay
        // the built-in replaceChildren()s the container on each load; the
        // overlay is re-injected here when it gets wiped
        if (!this.overlayCanvas || this.overlayCanvas.parentElement !== container) {
            this.overlayCanvas = document.createElement("canvas");
            this.overlayCanvas.style.cssText =
                "position:absolute;top:0;left:0;pointer-events:none;";
            container.appendChild(this.overlayCanvas);
        }
        return container;
    }

    /**
     * Pin the overlay to the video CONTENT's rendered rect. The video element
     * fills its box with objectFit: contain, so when the box aspect ratio
     * differs from the video's, the frames are letterboxed/centered — the
     * overlay must follow that exact rect or the mask drifts. The box is the
     * video element's own layout box (offsetWidth/Height, unaffected by the
     * canvas zoom) — the framework may size it in px, so we never assume it.
     * Recomputed every frame, so node drags/resizes stay aligned.
     */
    _syncOverlayRect() {
        const container = this.node.videoContainer;
        const video = container?.querySelector("video");
        if (!container || !video || !this.overlayCanvas) return;
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const boxW = video.offsetWidth;
        const boxH = video.offsetHeight;
        if (!vw || !vh || !boxW || !boxH) return;
        const scale = Math.min(boxW / vw, boxH / vh);
        const w = vw * scale;
        const h = vh * scale;
        this.overlayCanvas.style.left = (boxW - w) / 2 + "px";
        this.overlayCanvas.style.top = (boxH - h) / 2 + "px";
        this.overlayCanvas.style.width = w + "px";
        this.overlayCanvas.style.height = h + "px";
    }

    /** Pick up the built-in <video> and re-init the timeline when it changes. */
    _syncVideo() {
        const container = this.node.videoContainer;
        const video = container?.querySelector("video");
        if (!video) return;
        // fill the box with the same fit rule so the overlay lines up; the
        // waveform strip takes the bottom STRIP_H pixels of the box. A
        // percentage height keeps the video out of the container's layout
        // flow, so the framework's widget box (computeLayoutSize) owns the
        // height and the strip does not stretch the preview area.
        video.style.width = "100%";
        video.style.height = "calc(100% - " + STRIP_H + "px)";
        video.style.objectFit = "contain";
        video.style.display = "block";
        if (video !== this.lastVideoEl || video.currentSrc !== this.lastSrc) {
            this.lastVideoEl = video;
            this.lastSrc = video.currentSrc;
            this._initFromVideo(video);
        }
    }

    /**
     * The waveform strip: a canvas pinned to the bottom of the preview box.
     * The built-in video-preview widget's box is grown by STRIP_H so the
     * video area stays untouched (the framework owns the layout).
     */
    _ensureWaveform() {
        const container = this.node.videoContainer;
        if (!container) return;
        if (!this._sizeWrapped) {
            this._sizeWrapped = true;
            const w = this.node.widgets?.find((x) => x.name === "video-preview");
            if (w && !w.__lpvmeWrapped) {
                w.__lpvmeWrapped = true;
                const orig = w.computeLayoutSize;
                w.computeLayoutSize = () => {
                    const s = orig ? orig() : { minWidth: 200, minHeight: 64 };
                    return { ...s, minHeight: (s.minHeight || 0) + STRIP_H };
                };
            }
        }
        if (!this.stripCanvas || this.stripCanvas.parentElement !== container) {
            this.stripCanvas = document.createElement("canvas");
            this.stripCanvas.style.cssText =
                "position:absolute;left:0;right:0;bottom:0;width:100%;height:" +
                STRIP_H +
                "px;pointer-events:none;";
            container.appendChild(this.stripCanvas);
            this.lastPlayheadX = null;
        }
        return this.stripCanvas;
    }

    /** Load the decoded peaks when the video file changes. */
    async _syncWaveform() {
        const src = this.node.videoContainer?.querySelector("video")?.currentSrc;
        if (!src || src === this.lastAudioSrc) return;
        this.lastAudioSrc = src;
        this.audioPeaks = await loadAudioPeaks(src);
        // the tick may have drawn without peaks yet (no intervals); force a
        // redraw now that the waveform data is available
        this.lastPlayheadX = null;
    }

    /** Draw intervals (from the audio_mask widget) + the playhead marker. */
    _drawWaveform() {
        const canvas = this.stripCanvas;
        const video = this.node.videoContainer?.querySelector("video");
        if (!canvas || !video) return;
        const value = this.getAudioMaskWidget()?.value;
        if (value !== this.lastAudioMaskValue) {
            this.lastAudioMaskValue = value;
            this.audioIntervals = parseAudioIntervals(value);
        }
        const duration = this.audioPeaks?.duration || video.duration || 0;
        const playX = duration > 0 ? Math.round(((video.currentTime || 0) / duration) * canvas.clientWidth) : -1;
        // redraw when the playhead moves OR the intervals changed
        if (playX !== this.lastPlayheadX || value !== this.lastDrawnValue) {
            this.lastPlayheadX = playX;
            this.lastDrawnValue = value;
            drawWaveform(
                canvas,
                this.audioPeaks,
                duration,
                this.audioIntervals || [],
                video.currentTime || 0
            );
        }
    }

    getAudioMaskWidget() {
        return this.node.widgets?.find((w) => w.name === "audio_mask");
    }

    async _initFromVideo(video) {
        for (let tries = 0; tries < 50 && (!video.videoWidth || !video.videoHeight); tries++) {
            await new Promise((resolve) => setTimeout(resolve, 120));
        }
        this.fps = await measureVideoFps(video);
        this.frameCount = Math.max(1, Math.round((video.duration || 0) * this.fps));
        this.timeline = new MaskTimeline();
        this.timeline.setSize(video.videoWidth, video.videoHeight);
        this.lastMaskIdx = null;
        const filename = this._getVideoFilename();
        if (filename) {
            const metaLoaded = await this._autoLoadMaskMeta(filename);
            if (!metaLoaded) {
                await this._reloadKeyframes();
            }
        } else {
            await this._reloadKeyframes();
        }
        this.lastKeyframesValue = this._keyframesWidget()?.value;
    }

    /**
     * Fetch `/lanpaint/video_mask_meta` for the video and, if metadata is
     * present, upload the base64 keyframes as PNGs and populate the timeline
     * and widget. Returns true when metadata was found and loaded.
     */
    async _autoLoadMaskMeta(filename) {
        try {
            const url =
                api.apiURL(
                    "/lanpaint/video_mask_meta?filename=" + encodeURIComponent(filename)
                ) + "&rand=" + Math.random();
            const resp = await fetch(url);
            if (!resp.ok) throw new Error("HTTP " + resp.status);
            const json = await resp.json();
            if (!json.found || !json.payload) {
                // the video carries no mask metadata: clear the workflow's masks
                this.timeline.keyframes.clear();
                this.timeline.sdfCache.clear();
                const kw = this._keyframesWidget();
                if (kw) kw.value = "{}";
                const aw = this.getAudioMaskWidget();
                if (aw) aw.value = "[]";
                return false;
            }
            const payload = json.payload;
            this.timeline.keyframes.clear();
            this.timeline.sdfCache.clear();

            const keyframeData = payload.keyframes || {};
            const filenames = {};
            const stamp = Date.now();
            for (const [idxStr, b64png] of Object.entries(keyframeData)) {
                const idx = parseInt(idxStr, 10);
                if (isNaN(idx) || idx < 0 || idx >= this.frameCount) continue;
                try {
                    const img = await new Promise((resolve, reject) => {
                        const im = new Image();
                        im.onload = () => resolve(im);
                        im.onerror = () => reject(new Error("decode failed"));
                        im.src = "data:image/png;base64," + b64png;
                    });
                    const c = this.timeline.newMaskCanvas();
                    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
                    this.timeline.setKeyframe(idx, c);

                    const blob = await new Promise((r) => c.toBlob(r, "image/png"));
                    if (!blob) continue;
                    const name = "lanpaint_kf_" + stamp + "_" + idx + ".png";
                    const fd = new FormData();
                    fd.append("image", blob, name);
                    fd.append("type", "input");
                    const upResp = await api.fetchApi("/upload/image", {
                        method: "POST",
                        body: fd,
                    });
                    if (!upResp.ok) continue;
                    const upData = await upResp.json();
                    filenames[idx] = upData.name;
                } catch (e) {
                    console.warn("[LanPaint] preview auto-load keyframe decode failed:", e);
                }
            }

            const kw = this._keyframesWidget();
            if (kw) kw.value = JSON.stringify(filenames);
            const aw = this.getAudioMaskWidget();
            if (aw) aw.value = JSON.stringify(payload.audio_intervals || "[]");
            this._maskVideo = filename;

            if (this.node.widgets_values && this.node.widgets) {
                for (const ww of [kw, aw]) {
                    if (!ww) continue;
                    const wi = this.node.widgets.indexOf(ww);
                    if (wi >= 0) this.node.widgets_values[wi] = ww.value;
                }
            }

            console.log(
                "[LanPaint NodeMaskPreview] auto-loaded",
                Object.keys(filenames).length,
                "keyframes from mp4 metadata"
            );
            return true;
        } catch (err) {
            console.warn("[LanPaint] preview video_mask_meta fetch failed:", err);
            // no metadata could be read: treat like not-found and clear
            this.timeline.keyframes.clear();
            this.timeline.sdfCache.clear();
            const kw = this._keyframesWidget();
            if (kw) kw.value = "{}";
            const aw = this.getAudioMaskWidget();
            if (aw) aw.value = "[]";
            return false;
        }
    }

    async _reloadKeyframes() {
        const w = this._keyframesWidget();
        const timeline = this.timeline;
        if (!w || !timeline) return;
        let data = {};
        try {
            data = JSON.parse(w.value || "{}");
        } catch (_) {
            return;
        }
        timeline.keyframes.clear();
        timeline.sdfCache.clear();
        for (const [idxStr, filename] of Object.entries(data)) {
            const idx = parseInt(idxStr, 10);
            if (isNaN(idx) || idx < 0 || idx >= this.frameCount) continue;
            try {
                const url =
                    api.apiURL("/view?filename=" + encodeURIComponent(filename) + "&type=input") +
                    app.getRandParam();
                const img = await new Promise((resolve, reject) => {
                    const im = new Image();
                    im.crossOrigin = "anonymous";
                    im.onload = () => resolve(im);
                    im.onerror = () => reject(new Error("cannot load " + filename));
                    im.src = url;
                });
                const c = timeline.newMaskCanvas();
                c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
                timeline.setKeyframe(idx, c);
            } catch (err) {
                console.warn("[LanPaint VideoMaskEditor] preview keyframe load failed:", err);
            }
        }
        this.lastMaskIdx = null;
    }

    /** Draw only the red mask over the (already rendering) built-in <video>. */
    _drawOverlay() {
        const video = this.node.videoContainer?.querySelector("video");
        if (!video || !this.timeline || !this.overlayCanvas || video.readyState < 2) return;
        // keep the overlay bitmap at the video resolution (it is re-created
        // when the built-in replaceChildren()s its container)
        if (this.overlayCanvas.width !== this.timeline.width) {
            this.overlayCanvas.width = this.timeline.width;
            this.overlayCanvas.height = this.timeline.height;
        }
        // the browser presents frame floor(t*fps); round() mislabels the mask
        // by one whenever the video pauses mid-frame. floor with an epsilon
        // guard for float dust at exact frame times (t = I/fps * fps).
        const idx = Math.min(
            this.frameCount - 1,
            Math.max(0, Math.floor(video.currentTime * this.fps + 1e-6))
        );
        if (idx === this.lastMaskIdx) return; // nothing changed since last frame
        this.lastMaskIdx = idx;
        const ctx = this.overlayCanvas.getContext("2d");
        ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        const data = this.timeline.maskAt(idx);
        if (!data) return;
        // putImageData ignores compositing, so the mask travels through a
        // persistent canvas; source-in paints the red tint only where the
        // mask alpha is
        if (this._maskCanvas?.width !== this.overlayCanvas.width) {
            this._maskCanvas = this.timeline.newMaskCanvas();
        }
        this._maskCanvas.getContext("2d").putImageData(data, 0, 0);
        ctx.drawImage(this._maskCanvas, 0, 0);
        ctx.globalCompositeOperation = "source-in";
        ctx.fillStyle = "rgba(255, 40, 40, 0.75)";
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.globalCompositeOperation = "source-over";
    }

    _tick() {
        if (this.disposed) return;
        const kVal = this._keyframesWidget()?.value;
        if (kVal !== this.lastKeyframesValue) {
            this.lastKeyframesValue = kVal;
            this._reloadKeyframes();
        }
        const container = this._ensureOverlay();
        this._syncVideo();
        this._syncOverlayRect();
        if (container) this._drawOverlay();
        this._ensureWaveform();
        this._syncWaveform();
        this._drawWaveform();
        this.raf = requestAnimationFrame(() => this._tick());
    }

    dispose() {
        this.disposed = true;
        cancelAnimationFrame(this.raf);
        this.overlayCanvas = null;
        this.stripCanvas = null;
    }
}

/* ------------------------------------------------------------------ */
/* extension                                                           */
/* ------------------------------------------------------------------ */

app.registerExtension({
    name: "LanPaint.VideoMaskEditor",
    async nodeCreated(node) {
        if (!node?.comfyClass || node.comfyClass !== NODE_CLASS) {
            return;
        }

        // hide the keyframes and audio_mask data widgets from the node body.
        // The litegraph fork's draw/layout path skips widgets via the direct
        // `hidden` flag (isWidgetVisible), so set it in addition to
        // type/options.
        for (const wname of ["keyframes", "audio_mask"]) {
            const hw = node.widgets?.find((w) => w.name === wname);
            if (hw) {
                hw.type = "hidden";
                hw.hidden = true;
                hw.options = { ...(hw.options || {}), hidden: true };
                hw.draw = () => undefined;
                hw.computeSize = () => [0, -4];
            }
        }

        node.addWidget("button", "Edit Video Mask", null, () => {
            if (currentEditor) currentEditor.close();
            currentEditor = new VideoMaskEditor(node);
            currentEditor.open();
        });

        // live preview: the video with the mask overlay, available as soon as
        // a video file is picked (no run needed), updated when the editor
        // saves. Created LAST so it sits at the bottom of the node body, like
        // a video loader's player below its file widget.
        if (!node.__lpvmePreview) {
            node.__lpvmePreview = new NodeMaskPreview(node);
            node.onRemoved = () => {
                node.__lpvmePreview?.dispose();
                node.__lpvmePreview = null;
            };
        }
    },
});
