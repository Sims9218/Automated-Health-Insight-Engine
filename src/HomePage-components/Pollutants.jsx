import React from "react";
import "./PollutantsStyle.css";

function Pollutants(){
    return(
        <div className="PollutantsBox">
            <h1>Pollutants</h1>
            <div className="pollutant-grid">
                <div className="pollutant-item">CO</div>
                <div className="pollutant-item">NO</div>
                <div className="pollutant-item">NO2</div>
                <div className="pollutant-item">O3</div>
                <div className="pollutant-item">SO2</div>
                <div className="pollutant-item">PM2.5</div>
                <div className="pollutant-item">PM10</div>
                <div className="pollutant-item">NH3</div>
            </div>
        </div>
    );
}

export default Pollutants;