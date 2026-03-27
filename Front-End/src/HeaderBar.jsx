import React from "react";
import "./aqi-header.css";

function HeaderBar() {
  return (
    <header>
      <h1>Air-Quality</h1>

      <div id="header-button">
        <button className="btn no-margin">About</button>
        <button className="btn">More About</button>
      </div>
    </header>
  );
}

export default HeaderBar;
