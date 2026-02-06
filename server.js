import express from "express";

const app = express();
app.use(express.json());

app.post("/notify", (req, res) => {
  const { app, title, body, time } = req.body;

  // 🚫 Hard filters for system noise
  const blockedPhrases = [
    "doing work in the background",
    "checking for messages",
    "sync",
    "service",
    "running"
  ];

  if (
    !body ||
    blockedPhrases.some(p =>
      body.toLowerCase().includes(p)
    )
  ) {
    console.log("⛔ System notification blocked:", body);
    return res.json({ status: "ignored" });
  }

  console.log("✅ Message received:", {
    app,
    title,
    body,
    time
  });

  res.json({ status: "accepted" });
});


app.listen(3000, () => {
  console.log("Server running on port 3000");
});
