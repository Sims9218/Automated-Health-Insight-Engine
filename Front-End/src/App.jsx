import React, { useState } from "react";
import HeaderBar from "./HeaderBar";
import HomePage from "./HomePage";
import "./App.css";

// [ADDED] HRI metric to soft gradient color map
const METRIC_BG = {
    Good:      "rgba(34, 197, 94, 0.15)",
    Moderate:  "rgba(234, 179, 8, 0.15)",
    Poor:      "rgba(249, 115, 22, 0.15)",
    Unhealthy: "rgba(239, 68, 68, 0.15)",
    Severe:    "rgba(185, 28, 28, 0.15)",
    Hazardous: "rgba(124, 58, 237, 0.15)",
};

function App() {
    const [city, setCity] = useState("Mumbai");
    // [ADDED] hriMetric lifted up so App can apply gradient across entire page
    const [hriMetric, setHriMetric] = useState(null);

    const accentColor = hriMetric
        ? (METRIC_BG[hriMetric] || "rgba(37, 99, 235, 0.08)")
        : "rgba(37, 99, 235, 0.08)";

    // [ADDED] Dynamic gradient on the root .App div — covers header + content
    const bgStyle = {
        background: `linear-gradient(135deg, ${accentColor} 0%, #dbeafe 50%, #f1f5f9 100%)`,
        transition: "background 0.8s ease",
    };

    return (
        <div className="App" style={bgStyle}>
            <HeaderBar city={city} setCity={setCity} />
            {/* [ADDED] Pass setHriMetric down so HomePage can report metric up */}
            <HomePage city={city} onMetricLoaded={setHriMetric} />
        </div>
    );
}

export default App;
