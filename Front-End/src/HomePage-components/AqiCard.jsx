import React from "react";
import "./AqiCardStyle.css";
import { useEffect, useState } from "react";
import { getLatestHRI } from "../api";

const METRIC_COLORS = {
    Good:      "#22c55e",
    Moderate:  "#eab308",
    Poor:      "#f97316",
    Unhealthy: "#ef4444",
    Severe:    "#b91c1c",
    Hazardous: "#7c3aed",
};

const SCALE_BANDS = [
    { max: 100,      label: "Good",      color: "#22c55e" },
    { max: 200,      label: "Moderate",  color: "#eab308" },
    { max: 300,      label: "Poor",      color: "#f97316" },
    { max: 500,      label: "Unhealthy", color: "#ef4444" },
    { max: 750,      label: "Severe",    color: "#b91c1c" },
    { max: Infinity, label: "Hazardous", color: "#7c3aed" },
];

const SCALE_MAX = 750;

// [ADDED] Inline SVG icons — no emoji, no external dependency
const HumidityIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2C12 2 5 10 5 14a7 7 0 0 0 14 0c0-4-7-12-7-12z"/>
    </svg>
);

const WindIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M9.59 4.59A2 2 0 1 1 11 8H2"/>
        <path d="M12.59 19.41A2 2 0 1 0 14 16H2"/>
        <path d="M6.59 11.41A2 2 0 1 0 8 8H2"/>
    </svg>
);

const PrecipIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 17.58A5 5 0 0 0 18 8h-1.26A8 8 0 1 0 4 16.25"/>
        <line x1="8" y1="19" x2="8" y2="21"/>
        <line x1="8" y1="13" x2="8" y2="15"/>
        <line x1="16" y1="19" x2="16" y2="21"/>
        <line x1="16" y1="13" x2="16" y2="15"/>
        <line x1="12" y1="21" x2="12" y2="23"/>
        <line x1="12" y1="15" x2="12" y2="17"/>
    </svg>
);

function AqiCard({ city }) {
    const [data, setData] = useState(null);

    useEffect(() => {
        getLatestHRI(city).then(setData);
    }, [city]);

    const circleColor = data ? (METRIC_COLORS[data.metric] || "#22c55e") : "#22c55e";
    const hri = data ? Math.min(Math.round(data.hri), SCALE_MAX) : 0;
    const markerPct = (hri / SCALE_MAX) * 100;

    return (
        <div className="aqi-card">
            <div className="Left-side">
                <div className="aqi-top">
                    <div className="hri-circle" style={{ background: circleColor }}>
                        {data ? Math.round(data.hri) : "..."}
                    </div>
                    <div className="health-info">
                        <p>Health Risk Index</p>
                        <h2>{data ? data.metric : "..."}</h2>
                    </div>
                </div>

                {/* Scale bar */}
                <div className="scale-wrap">
                    <div className="scale-bar">
                        {SCALE_BANDS.map((band, i) => {
                            const prev = i === 0 ? 0 : SCALE_BANDS[i - 1].max;
                            const curr = Math.min(band.max, SCALE_MAX);
                            const width = ((curr - prev) / SCALE_MAX) * 100;
                            return (
                                <div
                                    key={band.label}
                                    className="scale-segment"
                                    style={{ width: `${width}%`, background: band.color }}
                                />
                            );
                        })}
                        {data && (
                            <div className="scale-marker" style={{ left: `${markerPct}%` }} />
                        )}
                    </div>
                    <div className="scale-labels">
                        <span>0</span>
                        <span>100</span>
                        <span>200</span>
                        <span>300</span>
                        <span>500</span>
                        <span>750+</span>
                    </div>
                </div>
            </div>

            <div className="Mid-side">
                <p className="card-label">AQI</p>
                <h3 className="card-value">
                    {data ? Math.round(data.pm2_5 * 10) : "..."}
                </h3>
            </div>

            <div className="Right-side">
                <p className="card-label">Weather</p>
                <h3 className="card-value">{data ? `${data.temp}°C` : "..."}</h3>

                {/* [UPDATED] SVG icons instead of emojis */}
                <div className="weather-sub">
                    <div className="weather-sub-item">
                        <HumidityIcon />
                        <span>{data ? `${data.humidity}%` : "..."}</span>
                    </div>
                    <div className="weather-sub-item">
                        <WindIcon />
                        <span>{data ? `${data.wind_speed} m/s` : "..."}</span>
                    </div>
                    <div className="weather-sub-item">
                        <PrecipIcon />
                        <span>{data ? `${data.precip} mm` : "..."}</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default AqiCard;
