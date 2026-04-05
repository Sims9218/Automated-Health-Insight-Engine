const BASE_URL = "https://automated-health-insight-engine-1368.onrender.com"; 

export const getLatestHRI = async (city) => {
  try {
    const response = await fetch(`${BASE_URL}/latest-hri/${city}`);

    if (!response.ok) {
      throw new Error("Failed to fetch data");
    }

    return await response.json();

  } catch (error) {
    console.error("API Error:", error);
    return null;
  }
};

export const getForecast = async (city) => {
  try {
    const response = await fetch(`${BASE_URL}/forecast/${city}`);

    if (!response.ok) {
      throw new Error("Failed to fetch forecast");
    }

    return await response.json();

  } catch (error) {
    console.error("Forecast Error:", error);
    return [];
  }
};