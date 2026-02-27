import express from "express";
import fetch from "node-fetch";

const app = express();
app.use(express.json());

// =======================
// CONFIG
// =======================
const ML_API_URL = "http://127.0.0.1:8001/predict";

// System phrases to ignore
const SYSTEM_PHRASES = [
  "doing work in the background",
  "checking for messages",
  "is doing work",
  "background activity",
  "syncing",
  "updating",
  "running in background",
  "notification listener",
  "accessibility service",
];

// =======================
// GLOBAL STATE (IMPORTANT)
// =======================
let lastServiceLog = 0;
let lastMessage = "";
let lastMessageTime = 0;

// =======================
// HELPERS
// =======================
function logServiceOnce(msg) {
  const now = Date.now();
  if (now - lastServiceLog > 10_000) {
    console.log("⚠️ Ignored system notification:", msg);
    lastServiceLog = now;
  }
}

function isDuplicate(msg) {
  const now = Date.now();
  if (msg === lastMessage && now - lastMessageTime < 5_000) {
    return true;
  }
  lastMessage = msg;
  lastMessageTime = now;
  return false;
}

// =======================
// ROUTE
// =======================
app.post("/notify", async (req, res) => {
  console.log("\n================ NEW NOTIFICATION ================");
  console.log("➡️ RAW PAYLOAD:", JSON.stringify(req.body, null, 2));

  try {
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

    // Normalize message text
    const messageText =
      body || text || message || bigText || title || "";

    console.log("📝 Extracted message text:", messageText);
    console.log("📋 Available fields:", { body, text, message, bigText, title });

    // -----------------------
    // HARD FILTERS
    // -----------------------
    if (!messageText.trim()) {
      console.log("❌ Empty message, ignored");
      return res.json({ status: "ignored", reason: "empty" });
    }

    const isSystemNotification =
      !title &&
      (messageText.includes(" is ") ||
        messageText.includes(" in the background") ||
        messageText.includes(" is running"));

    if (isSystemNotification) {
      console.log("⚠️ System notification detected");
      return res.json({ status: "ignored", reason: "system_notification" });
    }

    const lowerMessage = messageText.toLowerCase();
    const matchedPhrase = SYSTEM_PHRASES.find(p =>
      lowerMessage.includes(p)
    );

    if (matchedPhrase) {
      logServiceOnce(`"${messageText}" (matched "${matchedPhrase}")`);
      return res.json({ status: "ignored", reason: "system_phrase" });
    }

    if (isDuplicate(messageText)) {
      console.log("🔁 Duplicate message ignored");
      return res.json({ status: "ignored", reason: "duplicate" });
    }

    // -----------------------
    // VALID MESSAGE
    // -----------------------
    console.log("📩 Message received:", {
      appName,
      title,
      body: messageText,
      sender,
      timestamp,
    });

    // -----------------------
    // SEND TO ML
    // -----------------------
    const aiResult = await analyzeWithAI(messageText);

    // -----------------------
    // POST-PROCESSING
    // -----------------------
    let finalResult = aiResult;

    // Very short messages
    if (
      messageText.trim().length < 10 &&
      aiResult.label === "phishing"
    ) {
      console.log("🔧 Override: too short to be phishing");
      finalResult = {
        label: "safe",
        confidence: 0.95,
        override: "too_short",
      };
    }

    // Casual chat patterns
    const casualPatterns = [
      /^(hey|hi|hello|yo|sup|wassup)\b/i,
      /^(ok|okay|k|kk|thanks|thx|ty|lol|haha|sure)$/i,
      /^(yes|no|yeah|yep|nope|maybe|idk)$/i,
    ];

    if (
      casualPatterns.some(p => p.test(messageText)) &&
      aiResult.label === "phishing"
    ) {
      console.log("🔧 Override: casual message");
      finalResult = {
        label: "safe",
        confidence: 0.9,
        override: "casual_message",
      };
    }

    console.log("🧠 AI RESULT:", finalResult);

    // -----------------------
    // RESPONSE
    // -----------------------
    return res.json({
      status: "analyzed",
      source: appName,
      prediction: finalResult.label,
      confidence: finalResult.confidence,
      ...(finalResult.override && { override: finalResult.override }),
    });

  } catch (error) {
    // 🔥 IMPORTANT FIX: full error visibility
    console.error("❌ Error in /notify");
    console.error("Message:", error.message);
    console.error("Stack:", error.stack);

    return res.status(500).json({
      status: "error",
      message: error.message,
    });
  }
});

// =======================
// ML CALL
// =======================
async function analyzeWithAI(message) {
  console.log("➡️ Sending to ML API:", message);

  const response = await fetch(ML_API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: message }),
  });

  console.log("⬅️ ML API status:", response.status);

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`ML API failed: ${response.status} ${text}`);
  }

  return response.json();
}

// =======================
// START SERVER
// =======================
app.listen(3000, () => {
  console.log("🚀 Server running on http://localhost:3000");
});
