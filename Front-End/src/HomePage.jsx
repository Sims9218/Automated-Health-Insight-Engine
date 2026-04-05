import React, { useState } from "react";
import AqiCard from "./HomePage-components/AqiCard";
import "./HomePageSS.css";
import HriDisplay from "./HomePage-components/HriDisplayInfo";
import Forecast from "./HomePage-components/Forecast";
import Pollutants from "./HomePage-components/Pollutants";
import SuggestionBox from "./HomePage-components/SuggestionBox";

function HomePage({city}) {
    // [ADDED] Lift HRI API response up from HriDisplay
    // so both SuggestionBox and other components can access advice + hriLabel
    const [hriData, setHriData] = useState(null);

    return (
        <div>
            <div className="Top-Container">
                <AqiCard />
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
