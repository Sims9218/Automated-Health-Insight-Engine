import React, { useEffect, useState } from "react";
import "./HriDisplayStyle.css";

function HriDisplay() {
    const [data, setData] = useState(null);

    useEffect(() => {
        fetch("https://hri-backend.onrender.com/latest-hri") // 🔁 replace with your URL
            .then(res => res.json())
            .then(data => {
                console.log(data);
                setData(data);
            })
            .catch(err => console.error(err));
    }, []);

    return (
        <div className="HriBox">
            <h2>Air Quality in</h2>

            {/* City */}
            <h1>{data ? data.city || "Mumbai" : "Loading..."}</h1>

            {/* Optional: show HRI */}
            {data && (
                <p style={{ fontSize: "20px", marginTop: "10px" }}>
                    HRI: {data.hri || "N/A"}
                </p>
            )}
        </div>
    );
}

export default HriDisplay;
