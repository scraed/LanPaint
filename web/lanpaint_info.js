import { app } from "../../scripts/app.js";

const LANPAINT_GITHUB_URL = "https://github.com/scraed/LanPaint";

const TARGET_NODES = new Set([
    "LanPaint_KSampler",
    "LanPaint_KSamplerAdvanced",
    "LanPaint_SamplerCustom",
    "LanPaint_SamplerCustomAdvanced",
]);

app.registerExtension({
    name: "LanPaint.InfoLink",
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
