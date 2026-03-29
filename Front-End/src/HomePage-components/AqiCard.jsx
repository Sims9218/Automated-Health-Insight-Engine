import React from "react";
import "./AqiCardStyle.css";
import { useEffect, useState } from "react";
import { getLatestHRI } from "../api";

function AqiCard() {
  const [data, setData] = useState(null);

  useEffect(() => {
      getLatestHRI().then(res => {
          const selected = res.find(item => item.city === "Mumbai") || res[0];
          setData(selected);
      });
  }, []);


  return (
    <div className="aqi-card">
      {/* Top Section */}
      <div className="aqi-top">
        <div className="hri-circle">
          {data ? Math.round(data.hri / 10) : "..."}
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
          <h3>{data ? Math.round(data.pm2_5 * 10) : "..."}</h3>
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