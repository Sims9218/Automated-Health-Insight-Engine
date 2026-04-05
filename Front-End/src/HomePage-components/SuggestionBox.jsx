import React from "react";
import "./SuggestionBoxStyle.css";

// [UPDATED] Removed METRIC_COLORS bg/border overrides — they broke the dark card theme.
// Styling is now handled entirely by CSS to match other cards.

// Maps each advice layer key to a display label and icon
const LAYER_META = {
    air:      { label: "Air Quality", icon: "💨" },
    temp:     { label: "Temperature", icon: "🌡️" },
    uv:       { label: "UV Index",    icon: "☀️" },
    wind:     { label: "Wind",        icon: "🌬️" },
    precip:   { label: "Rain",        icon: "🌧️" },
    festival: { label: "Today",       icon: "🎉" },
};

function SuggestionBox({ hriLabel, advice }) {

    // Build renderable items — skip null layers
    const items = [];

    if (advice) {
        // Air layer is an object with .text
        if (advice.air?.text) {
            items.push({
                key: "air",
                ...LAYER_META.air,
                text: advice.air.text,
                extra: advice.air.mask ? "😷 N95 mask recommended" : null,
            });
        }
        // String layers
        for (const key of ["temp", "uv", "wind", "precip"]) {
            if (advice[key]) {
                items.push({ key, ...LAYER_META[key], text: advice[key], extra: null });
            }
        }
        // Festival layer is an object with .name and .advice
        if (advice.festival) {
            items.push({
                key: "festival",
                icon: LAYER_META.festival.icon,
                label: advice.festival.name,
                text: advice.festival.advice,
                extra: null,
            });
        }
    }

    // Loading state while API hasn't responded yet
    if (!advice) {
        return (
            <div className="Suggestion">
                <p className="Suggestion-loading">Loading suggestions...</p>
            </div>
        );
    }

    return (
        <div className="Suggestion">
            {/* Header */}
            <div className="Suggestion-header">
                <h1 className="Suggestion-title">
                    Suggestions
                    {hriLabel && (
                        <span className="Suggestion-label"> — {hriLabel}</span>
                    )}
                </h1>
            </div>

            {/* Advice items — only non-null layers render */}
            <div className="Suggestion-body">
                {items.map((item, idx) => (
                    <React.Fragment key={item.key}>
                        <div className="Suggestion-item">
                            <span className="Suggestion-item-icon">{item.icon}</span>
                            <div>
                                <p className="Suggestion-item-label">{item.label}</p>
                                <p className="Suggestion-item-text">{item.text}</p>
                                {item.extra && (
                                    <p className="Suggestion-item-extra">{item.extra}</p>
                                )}
                            </div>
                        </div>
                        {idx < items.length - 1 && (
                            <div className="Suggestion-divider" />
                        )}
                    </React.Fragment>
                ))}

                {items.length === 0 && (
                    <p className="Suggestion-item-text">
                        Conditions are comfortable. No specific precautions needed.
                    </p>
                )}
            </div>
        </div>
    );
}

export default SuggestionBox;
