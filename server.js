import express from "express";
import fetch from "node-fetch";

const app = express();
app.use(express.json());

const SYSTEM_PHRASES = [
  "doing work in the background",
  "checking for messages"
];

app.post("/notify", async (req, res) => {
  const { app: appName, title, body, time } = req.body;

  const isSystemNotification =
    !body ||
    body.trim().length < 3 ||
    SYSTEM_PHRASES.some(p =>
      body.toLowerCase().includes(p)
    );

  if (isSystemNotification) {
    // Silent ignore (no spam logs)
    return res.json({ status: "ignored" });
  }

  console.log("📩 Message received:", {
    app: appName,
    title,
    body,
    time
  });

  try {
    // 🧠 Call ML API
    const aiResult = await analyzeWithAI(body);

    console.log("🧠 AI RESULT:", aiResult);

    res.json({
      status: "analyzed",
      source: appName,
      prediction: aiResult.label,
      confidence: aiResult.confidence
    });

  } catch (error) {
    console.error("❌ AI service error:", error.message);

    res.status(500).json({
      status: "error",
      message: "AI analysis failed"
    });
  }
});

async function analyzeWithAI(message) {
  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: message })
  });

  return response.json();
}

app.listen(3000, () => {
  console.log("Server running on port 3000");
});
