import React, { useEffect, useState } from "react";
import "./ForecastStyle.css";
import { getForecast } from "../api";   // [FIXED] use api.js instead of hardcoded old URL
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer
} from "recharts";

function Forecast({ city }) {
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    // [FIXED] Was fetching from friend's old Render URL directly.
    // Now uses getForecast from api.js so it hits your own backend.
    fetch(`https://automated-health-insight-engine-1368.onrender.com/forecast/${city}`)
  .then(res => res.json()) // ignore the update
      .then(data => {
        console.log("Forecast:", data);

        const formatted = data
          .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
          .map(item => ({
            hour: item.timestamp.slice(11, 16),   // HH:MM
            hri:  item.hri
          }));

        setChartData(formatted);
      })
      .catch(err => console.error(err));
  }, [city]);

  return (
    <div className="Prediction">
      <h1>Next-Day Forecast</h1>  {/* [UPDATED] label — data is next-day, not 24hr rolling */}

      {chartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="hour" stroke="rgba(255,255,255,0.5)" />
            <YAxis stroke="rgba(255,255,255,0.5)" />
            <Tooltip
              contentStyle={{
                background: "rgba(15,32,39,0.9)",
                border: "none",
                borderRadius: "8px",
                color: "white"
              }}
            />
            <Line
              type="monotone"
              dataKey="hri"
              stroke="#4CAF50"
              strokeWidth={3}
              dot={{ r: 4, fill: "#4CAF50" }}
            />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
}

export default Forecast;
