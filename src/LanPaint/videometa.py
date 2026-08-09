"""Read and write LanPaint mask metadata in MP4 files via PyAV.

The metadata tag ``lanpaint-mask`` holds a UTF-8 JSON payload:
{
  "version": 1,
  "video": "<source video filename>",
  "fps": <number>,
  "keyframes": {"<frame_idx>": "<base64 PNG (no data: prefix)>", ...},
  "audio_intervals": [{"start": <float>, "end": <float>}, ...]
}

Keyframes are base64-encoded PNG masks (RGBA, alpha = mask).
"No mask" = tag absent (None); "empty mask" = tag present with ``{}`` keyframes.

PyAV is import-guarded: the module-level ``av`` is ``None`` when PyAV is
unavailable, and the read/write functions raise ``RuntimeError`` in that case.
"""

import json
import os
from typing import Any, Dict, Optional

try:
    import av

    _HAS_AV = True
except ImportError:
    av = None  # type: ignore[assignment]
    _HAS_AV = False


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------

_METADATA_KEY = "lanpaint-mask"
_PAYLOAD_VERSION = 1


def encode_payload(payload: dict) -> str:
    """Encode a payload dict as a compact JSON string (UTF-8)."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def decode_payload(raw: Optional[str]) -> Optional[dict]:
    """Decode a metadata value to a payload dict, or None on any error.

    ``raw`` may be ``None`` (tag absent), a JSON string, or something
    unexpected.  Malformed / missing input returns ``None`` gracefully.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


# ---------------------------------------------------------------------------
# Core read / write
# ---------------------------------------------------------------------------


def read_mask_metadata(path: str) -> Optional[dict]:
    """Read the ``lanpaint-mask`` metadata tag from an MP4 file.

    Returns the decoded payload dict, or ``None`` when the tag is absent or
    PyAV is unavailable.
    """
    if not _HAS_AV:
        raise RuntimeError(
            "PyAV (av) is required to read mask metadata but is not installed"
        )
    with av.open(path, "r") as container:
        raw = container.metadata.get(_METADATA_KEY, None)
    return decode_payload(raw)


def write_mask_metadata(
    input_path: str,
    output_path: str,
    payload: dict,
) -> None:
    """Remux *input_path* to *output_path*, attaching a ``lanpaint-mask``
    metadata tag.

    The video track is stream-copied (no re-encode) so the video content is
    preserved byte-for-byte.  Audio and other tracks are also preserved.

    The source file at *input_path* is **never** modified.

    Raises ``RuntimeError`` when PyAV is unavailable.
    """
    if not _HAS_AV:
        raise RuntimeError(
            "PyAV (av) is required to write mask metadata but is not installed"
        )

    json_str = encode_payload(payload)

    with av.open(input_path, "r") as in_container:
        # ``movflags=use_metadata_tags`` is MANDATORY — without it ffmpeg
        # silently drops custom metadata keys from the output.
        with av.open(
            output_path,
            "w",
            format="mp4",
            options={"movflags": "use_metadata_tags"},
        ) as out_container:
            # Copy any existing metadata (except our own key) from the source.
            for key, value in in_container.metadata.items():
                if key != _METADATA_KEY:
                    out_container.metadata[key] = value

            # Write the LanPaint payload.
            out_container.metadata[_METADATA_KEY] = json_str

            # Stream-copy every stream from the source.
            stream_map: Dict[int, Any] = {}
            for in_stream in in_container.streams:
                out_stream = out_container.add_stream_from_template(in_stream)
                stream_map[in_stream.index] = out_stream

            # Demux → mux every packet.
            for packet in in_container.demux():
                if packet.dts is None:
                    continue
                out_stream = stream_map[packet.stream.index]
                packet.stream = out_stream
                out_container.mux(packet)

    # The caller is responsible for the output file on disk.


# ---------------------------------------------------------------------------
# Helpers for server routes
# ---------------------------------------------------------------------------


def _unique_output_path(input_dir: str, base_name: str) -> str:
    """Return a non-clobbering output filename in *input_dir*.

    Appends ``_masked``, then ``_masked_2``, ``_masked_3``, ... until a name
    that does not already exist is found.

    Returns the bare filename (not the full path).
    """
    stem, ext = os.path.splitext(base_name)
    candidate = f"{stem}_masked{ext}"
    if not os.path.exists(os.path.join(input_dir, candidate)):
        return candidate
    n = 2
    while True:
        candidate = f"{stem}_masked_{n}{ext}"
        if not os.path.exists(os.path.join(input_dir, candidate)):
            return candidate
        n += 1


def export_mask_video_from_request(
    input_dir: str,
    filename: str,
    keyframes: dict,
    audio_intervals: list,
    fps: float,
) -> str:
    """Remux a source video with mask metadata, returning the new filename.

    Parameters
    ----------
    input_dir:
        The ComfyUI input directory (``folder_paths.get_input_directory()``).
    filename:
        The source video filename (relative to *input_dir*).
    keyframes:
        Dict of ``{frame_idx: base64_png_string}``.
    audio_intervals:
        List of ``{"start": float, "end": float}`` dicts.
    fps:
        The video frame rate.

    Returns
    -------
    The bare filename of the exported MP4 (in *input_dir*).
    """
    payload: Dict[str, Any] = {
        "version": _PAYLOAD_VERSION,
        "video": filename,
        "fps": fps,
        "keyframes": keyframes,
        "audio_intervals": audio_intervals,
    }

    src = os.path.join(input_dir, filename)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"source video not found: {src}")

    out_name = _unique_output_path(input_dir, filename)
    out_path = os.path.join(input_dir, out_name)
    write_mask_metadata(src, out_path, payload)
    return out_name


def register_routes(server) -> None:
    """Register the LanPaint video-mask metadata routes on a ComfyUI
    PromptServer instance.

    Call this from ``__init__.py`` when running inside ComfyUI (the ``server``
    module is only importable in that environment).  Safe to call multiple
    times — routes are registered once.
    """
    import aiohttp
    import folder_paths

    @server.routes.get("/lanpaint/video_mask_meta")
    async def video_mask_meta(request: aiohttp.web.Request) -> aiohttp.web.Response:
        filename = request.query.get("filename", "")
        if not filename:
            return aiohttp.web.json_response({"found": False})

        input_dir = folder_paths.get_input_directory()
        path = os.path.join(input_dir, filename)
        if not os.path.isfile(path):
            return aiohttp.web.json_response({"found": False})

        try:
            payload = read_mask_metadata(path)
        except Exception:
            payload = None

        if payload is None:
            return aiohttp.web.json_response({"found": False})
        return aiohttp.web.json_response({"found": True, "payload": payload})

    @server.routes.post("/lanpaint/export_mask_video")
    async def export_mask_video(request: aiohttp.web.Request) -> aiohttp.web.Response:
        try:
            body = await request.json()
        except Exception:
            return aiohttp.web.json_response(
                {"error": "invalid JSON body"}, status=400
            )

        filename = body.get("filename", "")
        if not filename:
            return aiohttp.web.json_response(
                {"error": "missing filename"}, status=400
            )

        keyframes = body.get("keyframes", {})
        if not isinstance(keyframes, dict):
            return aiohttp.web.json_response(
                {"error": "keyframes must be a dict"}, status=400
            )

        audio_intervals = body.get("audio_intervals", [])
        if not isinstance(audio_intervals, list):
            return aiohttp.web.json_response(
                {"error": "audio_intervals must be a list"}, status=400
            )

        fps = body.get("fps", 30.0)
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            return aiohttp.web.json_response(
                {"error": "fps must be a number"}, status=400
            )

        input_dir = folder_paths.get_input_directory()
        try:
            out_name = export_mask_video_from_request(
                input_dir, filename, keyframes, audio_intervals, fps
            )
        except FileNotFoundError as e:
            return aiohttp.web.json_response({"error": str(e)}, status=404)
        except RuntimeError as e:
            return aiohttp.web.json_response({"error": str(e)}, status=500)
        except Exception as e:
            return aiohttp.web.json_response(
                {"error": f"export failed: {e}"}, status=500
            )

        return aiohttp.web.json_response(
            {"filename": out_name, "path": os.path.join(input_dir, out_name)}
        )
