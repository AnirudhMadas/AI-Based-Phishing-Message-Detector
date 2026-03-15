import express from "express";
import fetch from "node-fetch";
import dotenv from "dotenv";
import rateLimit from "express-rate-limit";

dotenv.config();

const app = express();
app.use(express.json());

// =======================
// ENV VALIDATION
// =======================
const { ML_API_URL, API_KEY, PORT } = process.env;

if (!ML_API_URL) {
  throw new Error("❌ ML_API_URL not defined in environment variables");
}

if (!API_KEY) {
  throw new Error("❌ API_KEY not defined in environment variables");
}

// =======================
// RATE LIMITING
// =======================
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 200,
});
app.use(limiter);

// =======================
// SYSTEM FILTERS
// =======================
const SYSTEM_PHRASES = [
  "doing work in the background",
  "checking for messages",
  "background activity",
  "syncing",
  "updating",
  "running in background",
  "notification listener",
  "accessibility service",
];

let lastMessage = "";
let lastMessageTime = 0;

function isDuplicate(msg) {
  const now = Date.now();
  if (msg === lastMessage && now - lastMessageTime < 5000) {
    return true;
  }
  lastMessage = msg;
  lastMessageTime = now;
  return false;
}

// =======================
// HEALTH ROUTE
// =======================
app.get("/health", (req, res) => {
  res.json({ status: "ok", service: "phishing-backend" });
});

// =======================
// NOTIFY ROUTE (Main)
// =======================
app.post("/notify", async (req, res) => {
  try {
    console.log("📩 Incoming request:", req.body);

    // =======================
    // API KEY SECURITY
    // =======================
    if (req.headers["x-api-key"] !== API_KEY) {
      console.warn("⛔ Unauthorized attempt");
      return res.status(401).json({ error: "Unauthorized" });
    }

    const {
      app: appName,
      title,
      body,
      text,
      message,
      bigText,
      sender,
      timestamp,
    } = req.body;

    const messageText =
      body || text || message || bigText || title || "";

    if (!messageText.trim()) {
      return res.json({ status: "ignored", reason: "empty" });
    }

    const lowerMessage = messageText.toLowerCase();

    if (SYSTEM_PHRASES.some(p => lowerMessage.includes(p))) {
      return res.json({ status: "ignored", reason: "system_phrase" });
    }

    if (isDuplicate(messageText)) {
      return res.json({ status: "ignored", reason: "duplicate" });
    }

    // =======================
    // CALL ML SERVICE
    // =======================
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);

    const response = await fetch(ML_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: messageText }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) {
      throw new Error(`ML API failed: ${response.status}`);
    }

    const aiResult = await response.json();

    let finalLabel = aiResult.prediction;
    let confidence = aiResult.confidence;

    // Override for short text false positives
    if (messageText.length < 10 && finalLabel === "phishing") {
      finalLabel = "safe";
      confidence = 0.95;
    }

    console.log("🤖 ML Result:", finalLabel, confidence);

    return res.json({
      status: "analyzed",
      source: appName || "unknown",
      prediction: finalLabel,
      confidence,
      message: messageText,
    });

  } catch (error) {
    console.error("❌ Error in /notify:", error.message);

    return res.status(500).json({
      status: "error",
      message: "Internal server error",
    });
  }
});

// =======================
// START SERVER
// =======================
app.listen(PORT || 3000, () => {
  console.log(`🚀 Server running on port ${PORT || 3000}`);
});