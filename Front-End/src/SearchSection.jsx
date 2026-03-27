import React from "react";
import "./AqiSearchSection.css";

function SearchSection() {
  const handleSubmit = (e) => {
    e.preventDefault();
    const query = e.target.query.value;
    console.log("Search:", query);
  };

  return (
    <section>
      <div className="search-container">
        <form className="sc" onSubmit={handleSubmit}>
          <input
            id="search-box-input"
            type="text"
            placeholder="Search Location"
            name="query"
          />
          <div className="divider" />
          <button id="search-box-submit" type="submit">
            Search
          </button>
        </form>
      </div>
    </section>
  );
}

export default SearchSection;
