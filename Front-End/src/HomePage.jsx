import React, { useState, useEffect } from "react";
import AqiCard from "./HomePage-components/AqiCard";
import "./HomePageSS.css";
import Forecast from "./HomePage-components/Forecast";
import Pollutants from "./HomePage-components/Pollutants";
import SuggestionBox from "./HomePage-components/SuggestionBox";
import { getLatestHRI } from "./api";

// [ADDED] HRI metric to gradient color map for dynamic background
const METRIC_BG = {
    Good:      "rgba(34, 197, 94, 0.12)",    // soft green
    Moderate:  "rgba(234, 179, 8, 0.12)",    // soft yellow
    Poor:      "rgba(249, 115, 22, 0.12)",   // soft orange
    Unhealthy: "rgba(239, 68, 68, 0.12)",    // soft red
    Severe:    "rgba(185, 28, 28, 0.12)",    // soft dark red
    Hazardous: "rgba(124, 58, 237, 0.12)",   // soft purple
};

function HomePage({ city }) {
    const [hriData, setHriData] = useState(null);

    useEffect(() => {
        if (city) {
            getLatestHRI(city).then(setHriData);
        }
    }, [city]);

    // [ADDED] Pick accent color based on metric, fallback to default blue
    const accentColor = hriData?.metric
        ? METRIC_BG[hriData.metric] || "rgba(37, 99, 235, 0.08)"
        : "rgba(37, 99, 235, 0.08)";

    // [ADDED] Dynamic gradient — blends accent into the existing light blue bg
    const bgStyle = {
        background: `linear-gradient(135deg, ${accentColor} 0%, #dbeafe 50%, #f1f5f9 100%)`,
        minHeight: "100vh",
        transition: "background 0.8s ease",  // smooth transition on city change
    };

    return (
        // [ADDED] Dynamic background wraps entire page content
        <div style={bgStyle}>
            {/* Top row — AqiCard + SuggestionBox side by side */}
            <div className="Top-Container">
                <div className="Top-left">
                    <AqiCard city={city} />
                </div>

                {/* [UPDATED] SuggestionBox moved here from bottom */}
                <div className="Top-right">
                    <SuggestionBox
                        hriLabel={hriData?.metric}
                        advice={hriData?.advice}
                    />
                </div>
            </div>

            {/* Bottom row — Forecast + Pollutants unchanged */}
            <div className="Bottom-Container">
                <Forecast city={city} />
                <Pollutants city={city} />
            </div>
        </div>
    );
}

export default HomePage;
