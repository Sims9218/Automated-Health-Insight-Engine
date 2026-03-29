import React, { useEffect, useState } from "react";
import { getLatestHRI } from "../api";
import "./HriDisplayStyle.css";

function HriDisplay() {
    const [data, setData] = useState(null);

    useEffect(() => {
    getLatestHRI()
        .then(data => {
            console.log("API:", data);

            // 👇 FIX: pick one city from array
            const selected = data.find(item => item.city === "Mumbai") || data[0];

            setData(selected);
        })
        .catch(err => console.error(err));
}, []);

    return (
        <div className="HriBox">
            <h2>Air Quality in</h2>

            {/* City */}
            <h1>{data ? data.city || "Mumbai" : "Loading..."}</h1>

            {/* HRI */}
            {data && (
                <p style={{ fontSize: "20px", marginTop: "10px" }}>
                    HRI: {data.hri || "N/A"}
                </p>
            )}
        </div>
    );
}

export default HriDisplay;
