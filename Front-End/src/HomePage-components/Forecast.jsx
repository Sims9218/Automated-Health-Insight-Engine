import React, { useEffect, useState } from "react";
import "./ForecastStyle.css";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer
} from "recharts";

function Forecast({city}) {
  const [chartData, setChartData] = useState([]);

  useEffect(() => { 
    fetch(`https://automated-health-insight-engine-1368.onrender.com/forecast/${city}`)
      .then(res => res.json())
      .then(data => {
        console.log("Forecast:", data);

        const formatted = data
          .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp)) // sort
          .map(item => ({
            hour: item.timestamp.slice(11, 16), // HH:MM
            hri: item.hri
          }));

        setChartData(formatted);
      })
      .catch(err => console.error(err));
  }, [city]);

  return (
    <div className="Prediction">
      <h1>24-Hour Forecast</h1>

      {chartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" />
            <YAxis />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="hri"
              stroke="#4CAF50"
              strokeWidth={3}
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