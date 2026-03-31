import React, { useEffect, useState } from "react";
import { getLatestHRI } from "../api";
import "./HriDisplayStyle.css";

function HriDisplay({ city }) {
    const [data, setData] = useState(null);

    useEffect(() => {
    getLatestHRI(city)
        .then(data => {
            console.log("API:", data);
            setData(data);
        })
        .catch(err => console.error(err));
    }, [city]); 

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
