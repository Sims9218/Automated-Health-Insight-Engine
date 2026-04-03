import React from "react";
import "./SuggestionBoxStyle.css";

// [ADDED] Colour map per HRI label — used for the left accent border
const METRIC_COLORS = {
    Good:      { color: "#16a34a", bg: "#f0fdf4", border: "#bbf7d0" },
    Moderate:  { color: "#ca8a04", bg: "#fefce8", border: "#fef08a" },
    Poor:      { color: "#ea580c", bg: "#fff7ed", border: "#fed7aa" },
    Unhealthy: { color: "#dc2626", bg: "#fef2f2", border: "#fecaca" },
    Severe:    { color: "#9f1239", bg: "#fff1f2", border: "#fecdd3" },
    Hazardous: { color: "#581c87", bg: "#faf5ff", border: "#e9d5ff" },
};

const DEFAULT_THEME = { color: "#64748b", bg: "#f8fafc", border: "#e2e8f0" };

// [ADDED] Maps each advice layer key to a display label and icon
const LAYER_META = {
    air:     { label: "Air Quality", icon: "💨" },
    temp:    { label: "Temperature", icon: "🌡️" },
    uv:      { label: "UV Index",    icon: "☀️" },
    wind:    { label: "Wind",        icon: "🌬️" },
    precip:  { label: "Rain",        icon: "🌧️" },
    festival:{ label: "Today",       icon: "🎉" },
};

function SuggestionBox({ hriLabel, advice }) {
    // [ADDED] Pick theme based on current HRI label
    const theme = METRIC_COLORS[hriLabel] || DEFAULT_THEME;

    // [ADDED] Build renderable advice items — skip layers with null advice
    // 'air' layer is an object; others are plain strings or null; festival is object or null
    const items = [];

    if (advice) {
        // Air layer — object with .text field
        if (advice.air?.text) {
            items.push({
                key: "air",
                ...LAYER_META.air,
                text: advice.air.text,
                extra: advice.air.mask ? "😷 N95 mask recommended" : null,
            });
        }
        // String layers — temp, uv, wind, precip
        for (const key of ["temp", "uv", "wind", "precip"]) {
            if (advice[key]) {
                items.push({ key, ...LAYER_META[key], text: advice[key], extra: null });
            }
        }
        // Festival layer — object with .name and .advice
        if (advice.festival) {
            items.push({
                key: "festival",
                ...LAYER_META.festival,
                label: advice.festival.name,
                text: advice.festival.advice,
                extra: null,
            });
        }
    }

    // Show loading state while advice is being fetched
    if (!advice) {
        return (
            <div className="Suggestion" style={{ backgroundColor: "#f8fafc", borderLeft: "4px solid #e2e8f0" }}>
                <p className="Suggestion-loading">Loading suggestions...</p>
            </div>
        );
    }

    return (
        <div
            className="Suggestion"
            style={{
                // [ADDED] Dynamic accent border colour based on HRI level
                backgroundColor: theme.bg,
                borderLeft: `4px solid ${theme.color}`,
            }}
        >
            {/* Header */}
            <div className="Suggestion-header">
                <h2 className="Suggestion-title">
                    Suggestions
                    {hriLabel && (
                        <span className="Suggestion-label" style={{ color: theme.color }}>
                            {" "}— {hriLabel}
                        </span>
                    )}
                </h2>
            </div>

            {/* [ADDED] Render only layers that have actual advice (non-null) */}
            <div className="Suggestion-body">
                {items.map((item, idx) => (
                    <React.Fragment key={item.key}>
                        <div className="Suggestion-item">
                            <span className="Suggestion-item-icon">{item.icon}</span>
                            <div>
                                <p className="Suggestion-item-label">{item.label}</p>
                                <p className="Suggestion-item-text">{item.text}</p>
                                {/* [ADDED] Show mask badge only when air layer recommends it */}
                                {item.extra && (
                                    <p className="Suggestion-item-extra">{item.extra}</p>
                                )}
                            </div>
                        </div>
                        {/* Divider between items, not after last */}
                        {idx < items.length - 1 && (
                            <div className="Suggestion-divider" />
                        )}
                    </React.Fragment>
                ))}

                {/* Fallback if all layers are null */}
                {items.length === 0 && (
                    <p className="Suggestion-item-text" style={{ padding: "0 8px" }}>
                        Conditions are comfortable. No specific precautions needed.
                    </p>
                )}
            </div>
        </div>
    );
}

export default SuggestionBox;
