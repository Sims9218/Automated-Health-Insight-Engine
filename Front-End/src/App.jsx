import React, { useState } from "react";
import HeaderBar from "./HeaderBar";
import HomePage from "./HomePage";
import "./App.css";

function App() {
  return (
    <div className="App">
      <HeaderBar city={city} setCity={setCity} />
      <HomePage city={city} />
    </div>
  );
}

export default App;
