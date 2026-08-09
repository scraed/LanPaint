"""Tests for the LanPaint video mask metadata read/write (videometa module).

Requires PyAV.  All tests are skipped gracefully when PyAV is unavailable,
which is the case in CI environments (e.g., the system Python at C:\\Python314
does not have av installed, while the ComfyUI venv at E:\\ComfyUI\\.venv does).
"""

import os
import sys
import tempfile

import pytest

# Ensure the project root is on sys.path (tests may be run from any CWD).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import av
    import numpy as np

    HAVE_AV = True
except ImportError:
    av = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    HAVE_AV = False

pytestmark = pytest.mark.skipif(not HAVE_AV, reason="PyAV (av) not available")

if HAVE_AV:
    from src.LanPaint.videometa import (
        decode_payload,
        encode_payload,
        export_mask_video_from_request,
        read_mask_metadata,
        write_mask_metadata,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

if HAVE_AV:

    def _make_test_mp4(path: str, duration_frames: int = 10, fps: int = 10) -> None:
        """Create a tiny colour-bar MP4 via PyAV (no external tool required).

        The video is 32x24 pixels, RGB frames encoded with libx264.
        """
        container = av.open(path, "w", format="mp4")
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        for i in range(duration_frames):
            # simple colour bar: red channel varies per frame
            arr = np.zeros((24, 32, 3), dtype=np.uint8)
            r = (i * 25) % 256
            for py in range(24):
                for px in range(32):
                    arr[py, px] = [r, (px * 8) % 256, (py * 10) % 256]
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()


# ---------------------------------------------------------------------------
# Payload encode / decode
# ---------------------------------------------------------------------------


class TestPayloadCodec:
    def test_roundtrip_simple(self) -> None:
        payload = {
            "version": 1,
            "video": "test.mp4",
            "fps": 30.0,
            "keyframes": {"0": "iVBORw0KGgoAAAA="},
            "audio_intervals": [{"start": 1.0, "end": 2.5}],
        }
        encoded = encode_payload(payload)
        assert isinstance(encoded, str)
        decoded = decode_payload(encoded)
        assert decoded == payload

    def test_roundtrip_unicode(self) -> None:
        payload = {
            "version": 1,
            "video": "テスト動画.mp4",
            "fps": 24.0,
            "keyframes": {
                "0": "iVBORw0KGgoAAAA=",
                "12": "iVBORw0KGgoAAAANSUhEUg==",
            },
            "audio_intervals": [
                {"start": 0.5, "end": 1.25},
                {"start": 3.0, "end": 4.75},
            ],
        }
        encoded = encode_payload(payload)
        decoded = decode_payload(encoded)
        assert decoded == payload

    def test_empty_keyframes(self) -> None:
        payload = {
            "version": 1,
            "video": "no_mask.mp4",
            "fps": 30.0,
            "keyframes": {},
            "audio_intervals": [],
        }
        encoded = encode_payload(payload)
        decoded = decode_payload(encoded)
        assert decoded == payload

    def test_decode_none(self) -> None:
        assert decode_payload(None) is None

    def test_decode_garbage(self) -> None:
        assert decode_payload("not json") is None
        assert decode_payload(42) is None  # type: ignore[arg-type]
        assert decode_payload("[]") is None  # valid JSON but not a dict

    def test_decode_missing_keyframes_ok(self) -> None:
        # payloads missing optional fields still decode
        assert decode_payload('{"version":1,"video":"v.mp4"}') == {
            "version": 1,
            "video": "v.mp4",
        }


# ---------------------------------------------------------------------------
# Read / write round-trip
# ---------------------------------------------------------------------------


class TestWriteRead:
    def test_roundtrip_with_keyframes(self) -> None:
        payload = {
            "version": 1,
            "video": "src.mp4",
            "fps": 30.0,
            "keyframes": {
                "0": "iVBORw0KGgoAAAA=",
                "5": "iVBORw0KGgoAAAANSUhEUg==",
            },
            "audio_intervals": [{"start": 1.0, "end": 3.0}],
        }
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src.mp4")
            dst = os.path.join(td, "out.mp4")
            _make_test_mp4(src, duration_frames=10, fps=30)
            write_mask_metadata(src, dst, payload)
            assert os.path.isfile(dst)
            read_back = read_mask_metadata(dst)
            assert read_back == payload

    def test_tag_absent_on_plain_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "plain.mp4")
            _make_test_mp4(src)
            result = read_mask_metadata(src)
            assert result is None

    def test_source_is_never_modified(self) -> None:
        payload = {
            "version": 1,
            "video": "src.mp4",
            "fps": 25.0,
            "keyframes": {},
            "audio_intervals": [],
        }
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src.mp4")
            dst = os.path.join(td, "out.mp4")
            _make_test_mp4(src, duration_frames=5, fps=25)

            # hash the source bytes before writing
            src_bytes_before = open(src, "rb").read()
            write_mask_metadata(src, dst, payload)
            src_bytes_after = open(src, "rb").read()
            assert src_bytes_before == src_bytes_after, (
                "write_mask_metadata modified the source file"
            )

    def test_unicode_in_keyframe_data(self) -> None:
        # base64 strings can contain unicode context; metadata values are
        # UTF-8 ― ensure round-trip with non-ASCII frame indices works.
        payload = {
            "version": 1,
            "video": "видео.mp4",
            "fps": 24.0,
            "keyframes": {"0": "iVBORw0KGgoAAAA="},
            "audio_intervals": [],
        }
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src.mp4")
            dst = os.path.join(td, "out.mp4")
            _make_test_mp4(src, duration_frames=6, fps=24)
            write_mask_metadata(src, dst, payload)
            result = read_mask_metadata(dst)
            assert result is not None
            assert result["video"] == "видео.mp4"


# ---------------------------------------------------------------------------
# Filename suffixing (non-clobber)
# ---------------------------------------------------------------------------


class TestExportMaskVideo:
    def test_basic_export(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src_name = "video.mp4"
            src = os.path.join(td, src_name)
            _make_test_mp4(src, duration_frames=10, fps=30)
            out_name = export_mask_video_from_request(
                input_dir=td,
                filename=src_name,
                keyframes={"0": "abc123"},
                audio_intervals=[{"start": 0.0, "end": 2.0}],
                fps=30.0,
            )
            assert out_name == "video_masked.mp4"
            assert os.path.isfile(os.path.join(td, out_name))

    def test_suffix_when_target_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src_name = "video.mp4"
            src = os.path.join(td, src_name)
            _make_test_mp4(src, duration_frames=5, fps=25)

            # Create a fake collision file
            with open(os.path.join(td, "video_masked.mp4"), "wb") as f:
                f.write(b"not an mp4")

            out_name = export_mask_video_from_request(
                input_dir=td,
                filename=src_name,
                keyframes={},
                audio_intervals=[],
                fps=25.0,
            )
            assert out_name == "video_masked_2.mp4"
            assert os.path.isfile(os.path.join(td, out_name))

    def test_multiple_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src_name = "video.mp4"
            src = os.path.join(td, src_name)
            _make_test_mp4(src, duration_frames=5, fps=25)
            # create several collisions
            for name in ("video_masked.mp4", "video_masked_2.mp4", "video_masked_3.mp4"):
                with open(os.path.join(td, name), "wb") as f:
                    f.write(b"not an mp4")

            out_name = export_mask_video_from_request(
                input_dir=td,
                filename=src_name,
                keyframes={},
                audio_intervals=[],
                fps=25.0,
            )
            assert out_name == "video_masked_4.mp4"

    def test_source_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError):
                export_mask_video_from_request(
                    input_dir=td,
                    filename="nonexistent.mp4",
                    keyframes={},
                    audio_intervals=[],
                    fps=30.0,
                )

    def test_payload_fidelity(self) -> None:
        """The exported file must carry the exact payload that was given."""
        payload_kf = {"0": "iVBORw0KGgoAAAA=", "7": "/9j/4AAQSkZJRgABAQ=="}
        payload_ai = [
            {"start": 0.5, "end": 1.0},
            {"start": 3.25, "end": 5.75},
        ]
        with tempfile.TemporaryDirectory() as td:
            src_name = "vid.mp4"
            _make_test_mp4(os.path.join(td, src_name), duration_frames=10, fps=30)
            out_name = export_mask_video_from_request(
                input_dir=td,
                filename=src_name,
                keyframes=payload_kf,
                audio_intervals=payload_ai,
                fps=30.0,
            )
            payload = read_mask_metadata(os.path.join(td, out_name))
            assert payload is not None
            assert payload["version"] == 1
            assert payload["video"] == src_name
            assert payload["fps"] == 30.0
            assert payload["keyframes"] == payload_kf
            assert payload["audio_intervals"] == payload_ai


# ---------------------------------------------------------------------------
# Metadata coexistence
# ---------------------------------------------------------------------------


class TestMetadataCoexistence:
    def test_preexisting_metadata_is_preserved(self) -> None:
        """When the source MP4 already has metadata (e.g., title), it survives."""
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "labelled.mp4")
            # Create with a metadata tag
            container = av.open(src, "w", format="mp4")
            container.metadata["title"] = "Original Title"
            container.metadata["comment"] = "Should survive"
            stream = container.add_stream("libx264", rate=10)
            stream.width = 32
            stream.height = 24
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame.from_ndarray(
                np.zeros((24, 32, 3), dtype=np.uint8), format="rgb24"
            )
            for pkt in stream.encode(frame):
                container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()

            dst = os.path.join(td, "labelled_masked.mp4")
            payload = {
                "version": 1,
                "video": "labelled.mp4",
                "fps": 10.0,
                "keyframes": {},
                "audio_intervals": [],
            }
            write_mask_metadata(src, dst, payload)

            result = read_mask_metadata(dst)
            assert result == payload

            # Verify original metadata survived
            container2 = av.open(dst, "r")
            assert container2.metadata.get("title") == "Original Title"
            assert container2.metadata.get("comment") == "Should survive"
            container2.close()
