import React, { useState } from "react";
import AqiCard from "./HomePage-components/AqiCard";
import "./HomePageSS.css";
import HriDisplay from "./HomePage-components/HriDisplayInfo";
import Forecast from "./HomePage-components/Forecast";
import Pollutants from "./HomePage-components/Pollutants";
import SuggestionBox from "./HomePage-components/SuggestionBox";

function HomePage() {
    const [city, setCity] = useState("Mumbai");

    // [ADDED] Lift HRI API response up from HriDisplay
    // so both SuggestionBox and other components can access advice + hriLabel
    const [hriData, setHriData] = useState(null);

    return (
        <div>
            <div className="Top-Container">
                <div className="Top-left">
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

                    {/* [UPDATED] Pass onHriLoaded callback to capture full API response */}
                    <HriDisplay city={city} onHriLoaded={setHriData} />
                </div>

                <div className="Top-right">
                    <AqiCard city={city} />
                </div>
            </div>

            <div className="Bottom-Container">
                <Forecast city={city} />
                <Pollutants city={city} />
            </div>

            {/* [ADDED] Full-width suggestion strip — receives advice object from API */}
            <div className="Suggestion-Container">
                <SuggestionBox
                    hriLabel={hriData?.metric}
                    advice={hriData?.advice}
                />
            </div>
        </div>
    );
}

export default HomePage;
