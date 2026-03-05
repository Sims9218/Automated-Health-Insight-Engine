import React from "react";
import AqiCard from "./HomePage-components/AqiCard";
import "./HomePageSS.css"
import HriDisplay from "./HomePage-components/HriDisplayInfo";
import SuggestionBox from "./HomePage-components/SuggestionBox";
import Forecast from "./HomePage-components/Forecast";
import Pollutants from "./HomePage-components/Pollutants";

function HomePage(){
    return(
           <div>
                <div className="Top-Container">
                    <div className="Top-left">
                        <HriDisplay />
                    
                        <SuggestionBox />
                    </div>
                
                    <div className="Top-right">
                        <AqiCard />
                    </div> 
                </div>
                
                <div className="Bottom-Container">
                    <Forecast />
                    <Pollutants />
                </div>
           </div>
          
    );
}

export default HomePage;