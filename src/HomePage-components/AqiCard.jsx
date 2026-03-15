import React from "react";
import "./AqiCardStyle.css";

function AqiCard() {
  const hridata = {
    hrivalue: "5",
    hrimetric: "Healthy",
    aqidata: "150",
    weatherdata: "Sunny",
  };

  return (
    <div className="aqi-card">
      {/* Top Section */}
      <div className="aqi-top">
        <div className="hri-circle">
          {hridata.hrivalue}
        </div>

        <div className="health-info">
          <p>Health Risk Index</p>
          <h2>{hridata.hrimetric}</h2>
        </div>
      </div>

      {/* Divider */}
      <div className="aqi-divider"></div>

      {/* Bottom Section */}
      <div className="aqi-bottom">
        <div className="aqi-item">
          <p>AQI</p>
          <h3>{hridata.aqidata}</h3>
        </div>

        <div className="aqi-item">
          <p>Weather</p>
          <h3>{hridata.weatherdata}</h3>
        </div>
      </div>

    </div>
  );
}

export default AqiCard;