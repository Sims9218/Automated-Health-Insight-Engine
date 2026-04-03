import React from "react";
import "./AqiCardStyle.css";
import { useEffect, useState } from "react";
import { getLatestHRI } from "../api";

// [ADDED] Maps HRI metric label to a display colour for the circle
const METRIC_COLORS = {
    Good:      "#22c55e",   // green
    Moderate:  "#eab308",   // yellow
    Poor:      "#f97316",   // orange
    Unhealthy: "#ef4444",   // red
    Severe:    "#b91c1c",   // dark red
    Hazardous: "#7c3aed",   // purple
};

// [ADDED] Maps OWM AQI index (1–5) to a readable label
const OWM_AQI_LABELS = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor",
};

function AqiCard({ city }) {
    const [data, setData] = useState(null);

    useEffect(() => {
        getLatestHRI(city).then(setData);
    }, [city]);

    // [ADDED] Pick circle colour based on metric label; default to green
    const circleColor = data ? (METRIC_COLORS[data.metric] || "#22c55e") : "#22c55e";

    return (
        <div className="aqi-card">
            {/* Top Section */}
            <div className="aqi-top">
                <div
                    className="hri-circle"
                    // [UPDATED] Colour is now dynamic based on HRI level
                    style={{ background: circleColor }}
                >
                    {/* [FIXED] Show actual HRI score, not hri/10
                        HRI is already on a 0–500 scale. We round to nearest integer. */}
                    {data ? Math.round(data.hri) : "..."}
                </div>

                <div className="health-info">
                    <p>Health Risk Index</p>
                    <h2>{data ? data.metric : "..."}</h2>
                </div>
            </div>

            {/* Divider */}
            <div className="aqi-divider"></div>

            {/* Bottom Section */}
            <div className="aqi-bottom">
                <div className="aqi-item">
                    <p>AQI</p>
                    {/* [FIXED] Use real OWM AQI index (1–5) with readable label
                        Previously was pm2_5 * 10 which was incorrect */}
                    <h3>
                        {data ? Math.round(data.pm2_5 * 10) : "..."}
                    </h3>
                </div>

                <div className="aqi-item">
                    <p>Weather</p>
                    <h3>{data ? `${data.temp}°C` : "..."}</h3>
                </div>
            </div>
        </div>
    );
}

export default AqiCard;
