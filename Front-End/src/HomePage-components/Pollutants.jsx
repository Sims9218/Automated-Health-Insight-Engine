import React from "react";
import "./PollutantsStyle.css";
import { useEffect, useState } from "react";
import { getLatestHRI } from "../api";



function Pollutants(){
    const [data, setData] = useState(null);
    useEffect(() => {
    getLatestHRI().then(res => {
        const selected = res.find(item => item.city === "Mumbai") || res[0];
        setData(selected);
    });
}, []);
    return(
        <div className="PollutantsBox">
            <h1>Pollutants</h1>
            <div className="pollutant-grid">

                <div className="pollutant-item">
                    CO: {data ? data.co : "..."}
                </div>

                

                <div className="pollutant-item">
                    NO2: {data ? data.no2 : "..."}
                </div>

                <div className="pollutant-item">
                    O3: {data ? data.o3 : "..."}
                </div>

                
                <div className="pollutant-item">
                    PM2.5: {data ? data.pm2_5 : "..."}
                </div>

                <div className="pollutant-item">
                    PM10: {data ? data.pm10 : "..."}
                </div>

                
            </div>
        </div>
    );
}

export default Pollutants;