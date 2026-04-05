import React, { useState, useEffect } from "react";  // ← add useEffect
import AqiCard from "./HomePage-components/AqiCard";
import "./HomePageSS.css";
import HriDisplay from "./HomePage-components/HriDisplayInfo";
import Forecast from "./HomePage-components/Forecast";
import Pollutants from "./HomePage-components/Pollutants";
import SuggestionBox from "./HomePage-components/SuggestionBox";
import { getLatestHRI } from "./api"; // ← adjust path if needed

function HomePage({ city }) {
    const [hriData, setHriData] = useState(null);
    useEffect(() => {
        const fetchHRI = async () => {
            const data = await getLatestHRI(city);
            setHriData(data);
        };

        if (city) {
            fetchHRI();
        }
    }, [city]);

    return (
        <div>
            <div className="Top-Container">
                <AqiCard city={city}/>
            </div>

            <div className="Bottom-Container">
                <Forecast city={city} />
                <Pollutants city={city} />
            </div>

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
