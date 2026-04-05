import React, { useState, useEffect } from "react";
import AqiCard from "./HomePage-components/AqiCard";
import "./HomePageSS.css";
import Forecast from "./HomePage-components/Forecast";
import Pollutants from "./HomePage-components/Pollutants";
import SuggestionBox from "./HomePage-components/SuggestionBox";
import { getLatestHRI } from "./api";

// [UPDATED] onMetricLoaded callback added — reports HRI metric up to App for gradient
function HomePage({ city, onMetricLoaded }) {
    const [hriData, setHriData] = useState(null);

    useEffect(() => {
        if (city) {
            getLatestHRI(city).then(data => {
                setHriData(data);
                // [ADDED] Tell App what the current metric is so it can update gradient
                if (data?.metric && onMetricLoaded) {
                    onMetricLoaded(data.metric);
                }
            });
        }
    }, [city]);

    return (
        <div>
            <div className="Top-Container">
                <div className="Top-left">
                    <AqiCard city={city} />
                </div>
                <div className="Top-right">
                    <SuggestionBox
                        hriLabel={hriData?.metric}
                        advice={hriData?.advice}
                    />
                </div>
            </div>

            <div className="Bottom-Container">
                <Forecast city={city} />
                <Pollutants city={city} />
            </div>
        </div>
    );
}

export default HomePage;
