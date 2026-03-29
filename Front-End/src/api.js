const BASE_URL = "https://YOUR-RENDER-URL.onrender.com"; // 🔁 replace this

export const getLatestHRI = async () => {
  try {
    const response = await fetch(`${BASE_URL}/latest-hri`);

    if (!response.ok) {
      throw new Error("Failed to fetch data");
    }

    const data = await response.json();
    return data;

  } catch (error) {
    console.error("API Error:", error);
    return null;
  }
};
