import React from "react";
import "./aqi-header.css";

function HeaderBar() {
  const [city, setCity] = useState("Mumbai");
  return (
    <header>
      <h1>Air-Quality</h1>
      
      <div id="header-button">
            <div className="dropdown-container">
                        <select value={city} onChange={(e) => setCity(e.target.value)}>
                            <option value="" disabled>Choose City</option>
                            <option value="Mumbai">Mumbai</option>
                            <option value="Delhi">Delhi</option>
                            <option value="Hyderabad">Hyderabad</option>
                            <option value="Bangalore">Bangalore</option>
                            <option value="Pune">Pune</option>
                        </select>
                    </div>

        <button className="btn no-margin">About</button>
        <button className="btn">More About</button>
      </div>
    </header>
  );
}

export default HeaderBar;
