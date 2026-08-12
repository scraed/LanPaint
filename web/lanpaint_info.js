import { app } from "../../scripts/app.js";

const LANPAINT_GITHUB_URL = "https://github.com/scraed/LanPaint";

const TARGET_NODES = new Set([
    "LanPaint_KSampler",
    "LanPaint_KSamplerAdvanced",
    "LanPaint_SamplerCustom",
    "LanPaint_SamplerCustomAdvanced",
]);

// ---------------------------------------------------------------------------
// Key-based widgets_values migration.
//
// ComfyUI's graph format stores widget values as positional arrays, which
// silently corrupt when parameters are removed mid-list (a retired float
// would land in a combo slot and reset the kept parameters to defaults).
// LanPaint keeps a table of every historical layout (widget count -> widget
// names in order). On load, an old array is converted to a name-keyed map
// and re-emitted in the current widget order, so kept parameters keep their
// values by NAME and retired ones (Beta, Friction, ...) simply drop out.
// ---------------------------------------------------------------------------
// Layout rows name every historical widget position. The trailing
// "lanpaint_star_button" entry is the info-button widget this extension
// appends to every LanPaint node; it is serialized into widgets_values like
// any other widget and must occupy a slot in the layout.
const BUTTON = "lanpaint_star_button";
const LAYOUTS = {
    "LanPaint_KSampler": {
        current: [
            "LanPaint_NumSteps",
            "LanPaint_PromptMode",
            "LanPaint_Info",
            "Inpainting_mode",
        ],
        // 6 entries: 2.0.1 params + button; 5 entries: 2.0.0 params + button
        // (or 2.0.1 without the button) -- kept positions are identical.
        6: [
            "LanPaint_NumSteps",
            "LanPaint_PromptMode",
            "LanPaint_Info",
            "Inpainting_mode",
            "LanPaint_MinStepFrac",
            BUTTON,
        ],
        5: [
            "LanPaint_NumSteps",
            "LanPaint_PromptMode",
            "LanPaint_Info",
            "Inpainting_mode",
            BUTTON,
        ],
    },
    "LanPaint_KSamplerAdvanced": {
        current: [
            "LanPaint_NumSteps",
            "LanPaint_Lambda",
            "LanPaint_StepSize",
            "LanPaint_PromptMode",
            "LanPaint_Info",
            "Inpainting_mode",
        ],
        // 13 entries: 2.0.1 params + button; 12 entries: 2.0.0 params +
        // button (or 2.0.1 without the button) -- kept positions identical.
        13: [
            "LanPaint_NumSteps",
            "LanPaint_Lambda",
            "LanPaint_StepSize",
            "LanPaint_Beta",
            "LanPaint_Friction",
            "LanPaint_PromptMode",
            "LanPaint_EarlyStop",
            "LanPaint_Info",
            "Inpainting_mode",
            "LanPaint_InnerThreshold",
            "LanPaint_InnerPatience",
            "LanPaint_MinStepFrac",
            BUTTON,
        ],
        12: [
            "LanPaint_NumSteps",
            "LanPaint_Lambda",
            "LanPaint_StepSize",
            "LanPaint_Beta",
            "LanPaint_Friction",
            "LanPaint_PromptMode",
            "LanPaint_EarlyStop",
            "LanPaint_Info",
            "Inpainting_mode",
            "LanPaint_InnerThreshold",
            "LanPaint_InnerPatience",
            BUTTON,
        ],
    },
    "LanPaint_SamplerCustomAdvanced": {
        current: [
            "LanPaint_NumSteps",
            "LanPaint_Lambda",
            "LanPaint_StepSize",
            "LanPaint_PromptMode",
            "LanPaint_Info",
        ],
        // 12 entries: 2.0.1 params + button; 11 entries: 2.0.0 params +
        // button (or 2.0.1 without the button) -- kept positions identical.
        12: [
            "LanPaint_NumSteps",
            "LanPaint_Lambda",
            "LanPaint_StepSize",
            "LanPaint_Beta",
            "LanPaint_Friction",
            "LanPaint_PromptMode",
            "LanPaint_EarlyStop",
            "LanPaint_Info",
            "LanPaint_InnerThreshold",
            "LanPaint_InnerPatience",
            "LanPaint_MinStepFrac",
            BUTTON,
        ],
        11: [
            "LanPaint_NumSteps",
            "LanPaint_Lambda",
            "LanPaint_StepSize",
            "LanPaint_Beta",
            "LanPaint_Friction",
            "LanPaint_PromptMode",
            "LanPaint_EarlyStop",
            "LanPaint_Info",
            "LanPaint_InnerThreshold",
            "LanPaint_InnerPatience",
            BUTTON,
        ],
    },
};

function migrateWidgetsValues(nodeData) {
    const table = LAYOUTS[nodeData.type];
    if (!table || !Array.isArray(nodeData.widgets_values)) {
        return;
    }
    const wv = nodeData.widgets_values;
    const current = table.current;
    if (wv.length === current.length) {
        return; // already the current layout
    }
    const old = table[wv.length];
    if (!old) {
        return; // unknown layout: leave the array untouched
    }
    const byName = {};
    for (let i = 0; i < wv.length; i++) {
        byName[old[i]] = wv[i];
    }
    if (!current.every((name) => name in byName)) {
        return; // a kept parameter has no source value: leave as-is
    }
    nodeData.widgets_values = current.map((name) => byName[name]);
}

function installWidgetMigration() {
    const LGraphClass = app.graph?.constructor;
    if (!LGraphClass || LGraphClass.prototype.__lanpaint_migrated) {
        return;
    }
    LGraphClass.prototype.__lanpaint_migrated = true;

    const origConfigure = LGraphClass.prototype.configure;
    LGraphClass.prototype.configure = function (graphData, ...rest) {
        if (graphData && Array.isArray(graphData.nodes)) {
            for (const nodeData of graphData.nodes) {
                migrateWidgetsValues(nodeData);
            }
            const subgraphs = graphData.definitions?.subgraphs;
            if (Array.isArray(subgraphs)) {
                for (const sub of subgraphs) {
                    if (Array.isArray(sub?.nodes)) {
                        for (const nodeData of sub.nodes) {
                            migrateWidgetsValues(nodeData);
                        }
                    }
                }
            }
        }
        return origConfigure.call(this, graphData, ...rest);
    };
}

app.registerExtension({
    name: "LanPaint.InfoLink",
    setup() {
        installWidgetMigration();
    },
    async nodeCreated(node) {
        if (!node?.comfyClass || !TARGET_NODES.has(node.comfyClass)) {
            return;
        }

        const alreadyAdded = node.widgets?.some(
            (widget) => widget.name === "lanpaint_star_button"
        );
        if (alreadyAdded) {
            return;
        }

        node.addWidget(
            "button",
            "More Info, Bug Report, Star on GitHub ⭐",
            "lanpaint_star_button",
            () => {
                window.open(
                    LANPAINT_GITHUB_URL,
                    "_blank",
                    "noopener,noreferrer"
                );
            }
        );
    },
});
