"""Tests for the retired LanPaint hyperparameters and the value sanitizer.

Beta/Friction/EarlyStop/InnerThreshold/InnerPatience/MinStepFrac were
removed from the sampler node widgets; old prompts still pass them, so the
nodes accept-and-ignore them via hidden inputs. Invalid widget values fall
back to defaults instead of crashing the node.
"""

RETIRED = [
    "LanPaint_Beta",
    "LanPaint_Friction",
    "LanPaint_EarlyStop",
    "LanPaint_InnerThreshold",
    "LanPaint_InnerPatience",
    "LanPaint_MinStepFrac",
]


def _import_nodes():
    import LanPaint.src.LanPaint.nodes as nodes  # type: ignore[attr-defined]

    return nodes


def _required(node_cls):
    return node_cls.INPUT_TYPES().get("required", {})


def _hidden(node_cls):
    return node_cls.INPUT_TYPES().get("hidden", {})


def test_retired_params_removed_from_widgets() -> None:
    nodes = _import_nodes()
    for cls in (
        nodes.LanPaint_KSampler,
        nodes.LanPaint_KSamplerAdvanced,
        nodes.LanPaint_SamplerCustom,
        nodes.LanPaint_SamplerCustomAdvanced,
    ):
        req = _required(cls)
        for name in RETIRED:
            assert name not in req, f"{cls.__name__} still exposes {name}"


def test_retired_params_kept_as_hidden_inputs() -> None:
    nodes = _import_nodes()
    # The two advanced nodes exposed all six params; the basic KSampler
    # only ever had MinStepFrac. Old prompts must still validate.
    assert set(_hidden(nodes.LanPaint_KSamplerAdvanced)) >= set(RETIRED)
    assert set(_hidden(nodes.LanPaint_SamplerCustomAdvanced)) >= set(RETIRED)
    assert "LanPaint_MinStepFrac" in _hidden(nodes.LanPaint_KSampler)


def test_kept_widgets_still_present() -> None:
    nodes = _import_nodes()
    req = _required(nodes.LanPaint_KSamplerAdvanced)
    for name in (
        "LanPaint_NumSteps",
        "LanPaint_Lambda",
        "LanPaint_StepSize",
        "LanPaint_PromptMode",
        "LanPaint_Info",
        "Inpainting_mode",
    ):
        assert name in req


def test_sanitize_param_combos() -> None:
    nodes = _import_nodes()
    sanitize = nodes._sanitize_param
    allowed = ("Image First", "Prompt First")
    assert sanitize("Image First", "Image First", allowed=allowed) == "Image First"
    assert sanitize("Prompt First", "Image First", allowed=allowed) == "Prompt First"
    assert sanitize(1.0, "Image First", allowed=allowed) == "Image First"  # retired float
    assert sanitize("bogus", "Image First", allowed=allowed) == "Image First"
    assert sanitize(None, "Image First", allowed=allowed) == "Image First"


def test_sanitize_param_numbers() -> None:
    nodes = _import_nodes()
    sanitize = nodes._sanitize_param
    assert sanitize(5, 5) == 5
    assert sanitize(3.7, 0.2) == 3.7
    assert sanitize("abc", 0.2) == 0.2
    assert sanitize(None, 0.2) == 0.2
    assert sanitize(True, 5) == 5  # bool is not a valid number
