import React, { useEffect, useState } from "react";
import { getLatestHRI } from "../api";
import "./HriDisplayStyle.css";

function HriDisplay({ city, onHriLoaded }) {
    const [data, setData] = useState(null);

    useEffect(() => {
        getLatestHRI(city)
            .then(d => {
                console.log("API:", d);
                setData(d);
                // [ADDED] Notify parent (HomePage) once HRI data is loaded
                // so HomePage can pass hriLabel and advice down to SuggestionBox
                if (onHriLoaded && d) {
                    onHriLoaded(d);
                }
            })
            .catch(err => console.error(err));
    }, [city]);

    return (
        <div className="HriBox">
            <h2>Air Quality in</h2>

            {/* City name */}
            <h1>{data ? data.city : "Loading..."}</h1>

            {/* [UNCHANGED] HRI score display */}
            {data && (
                <p style={{ fontSize: "20px", marginTop: "10px" }}>
                    HRI: {data.hri ?? "N/A"}
                </p>
            )}

            {/* [ADDED] Show metric label below HRI score for clarity */}
            {data && data.metric && (
                <p style={{ fontSize: "14px", opacity: 0.75, marginTop: "4px" }}>
                    {data.metric}
                </p>
            )}
        </div>
    );
}

export default HriDisplay;
