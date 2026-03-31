import React, { useState } from "react";
import AqiCard from "./HomePage-components/AqiCard";
import "./HomePageSS.css"
import HriDisplay from "./HomePage-components/HriDisplayInfo";
import Forecast from "./HomePage-components/Forecast";
import Pollutants from "./HomePage-components/Pollutants";

function HomePage(){

    const [city, setCity] = useState("Mumbai"); // 
    return(
           <div>
                <div className="Top-Container">
                    <div className="Top-left">
                        <div className="dropdown-container">
                            <label>Choose City</label>
                            <select value={city} onChange={(e) => setCity(e.target.value)}>
                                <option value="Mumbai">Mumbai</option>
                                <option value="Delhi">Delhi</option>
                                <option value="Hyderabad">Hyderabad</option>
                                <option value="Bangalore">Bangalore</option>
                                <option value="Pune">Pune</option>
                            </select>
                        </div>
                        <HriDisplay city={city} /> {/*  pass city */}
                        
                    </div>
                
                    <div className="Top-right">
                        <AqiCard city={city} /> 
                    </div> 
                </div>
                
                <div className="Bottom-Container">
                    <Forecast city={city} /> {/* */}
                    <Pollutants city={city} /> {/*  */}
                </div>
           </div>
    );
}

export default HomePage;