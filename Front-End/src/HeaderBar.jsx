import React,{useState} from "react";
import "./aqi-header.css";

function HeaderBar({ city, setCity }) {

  return (
    <header>
      <h1>Air-Quality</h1>
      
      <div id="header-button">
            <div className="dropdown-container">
                        <select value={city} onChange={(e) => setCity(e.target.value)}>
                            <option value="Mumbai">Mumbai</option>
                            <option value="Delhi">Delhi</option>
                            <option value="Hyderabad">Hyderabad</option>
                            <option value="Bangalore">Bangalore</option>
                            <option value="Pune">Pune</option>
                        </select>
                    </div>

        <button className="btn no-margin">About</button>
      </div>
    </header>
  );
}

export default HeaderBar;
