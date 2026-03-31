import React, { useState } from "react";
import AqiCard from "./HomePage-components/AqiCard";
import "./HomePageSS.css"
import HriDisplay from "./HomePage-components/HriDisplayInfo";
import SuggestionBox from "./HomePage-components/SuggestionBox";
import Forecast from "./HomePage-components/Forecast";
import Pollutants from "./HomePage-components/Pollutants";

function HomePage(){

    const [city, setCity] = useState("Mumbai"); // 
    return(
           <div>
                <div style={{ margin: "10px" }}>
                    <select value={city} onChange={(e) => setCity(e.target.value)}>
                        <option value="Mumbai">Mumbai</option>
                        <option value="Delhi">Delhi</option>
                        <option value="Hyderabad">Hyderabad</option>
                    </select>
                </div>

                <div className="Top-Container">
                    <div className="Top-left">
                        <HriDisplay city={city} /> {/*  pass city */}
                        <SuggestionBox />
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