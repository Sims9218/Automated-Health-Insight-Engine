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

// [FIXED] Scale bands matching 0-60, 60-120, 120-180, 180-240, 240-300, 300+
const SCALE_BANDS = [
    { min: 0,   max: 60,       label: "Good",      color: "#22c55e" },
    { min: 60,  max: 120,      label: "Moderate",  color: "#eab308" },
    { min: 120, max: 180,      label: "Poor",      color: "#f97316" },
    { min: 180, max: 240,      label: "Unhealthy", color: "#ef4444" },
    { min: 240, max: 300,      label: "Severe",    color: "#b91c1c" },
    { min: 300, max: Infinity, label: "Hazardous", color: "#7c3aed" },
];

const SCALE_MAX = 300; // [FIXED] cap at 300 since 300+ is last band

// [FIXED] SVG icons with explicit fill="none" to prevent pink fill from global CSS
const HumidityIcon = () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{flexShrink:0}}>
        <path d="M12 2C12 2 5 10 5 14a7 7 0 0 0 14 0c0-4-7-12-7-12z" fill="none"/>
    </svg>
);

const WindIcon = () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{flexShrink:0}}>
        <path d="M9.59 4.59A2 2 0 1 1 11 8H2" fill="none"/>
        <path d="M12.59 19.41A2 2 0 1 0 14 16H2" fill="none"/>
        <path d="M6.59 11.41A2 2 0 1 0 8 8H2" fill="none" />
    </svg>
);

const PrecipIcon = () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{flexShrink:0}}>
        <path d="M20 17.58A5 5 0 0 0 18 8h-1.26A8 8 0 1 0 4 16.25" fill="none"/>
        <line x1="8"  y1="19" x2="8"  y2="21"/>
        <line x1="8"  y1="13" x2="8"  y2="15"/>
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

    // [FIXED] Clamp to SCALE_MAX for marker, position is % of 300
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
                        <p>Health Risk Index of {city}</p>
                        <h2>{data ? data.metric : "..."}</h2>
                    </div>
                </div>

                {/* Scale bar */}
                <div className="scale-wrap">
                    <div className="scale-bar">
                        {SCALE_BANDS.map((band, i) => {
                            // Each band is equal width since bands are 60 units each (300/5 = 60, last is 300+)
                            const width = (1 / SCALE_BANDS.length) * 100;
                            return (
                                <div
                                    key={band.label}
                                    className={`scale-segment ${i === 0 ? 'first' : ''} ${i === SCALE_BANDS.length - 1 ? 'last' : ''}`}
                                    style={{ width: `${width}%`, background: band.color }}
                                />
                            );
                        })}
                        {data && (
                            <div
                                className="scale-marker"
                                style={{ left: `calc(${markerPct}% - 1px)` }}
                            />
                        )}
                    </div>
                    <div className="scale-labels">
                        <span>0</span>
                        <span>60</span>
                        <span>120</span>
                        <span>180</span>
                        <span>240</span>
                        <span>300+</span>
                    </div>
                </div>
            </div>

            {/* AQI */}
            <div className="Mid-side">
                <p className="card-label">AQI</p>
                <h3 className="card-value">
                    {data ? Math.round(data.pm2_5 * 10) : "..."}
                </h3>
            </div>

            {/* Weather */}
            <div className="Right-side">
                <p className="card-label">Weather</p>
                <h3 className="card-value">{data ? `${data.temp}°C` : "..."}</h3>
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
