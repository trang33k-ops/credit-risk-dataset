let history = [];

const messages = document.getElementById("messages");
const input = document.getElementById("input");
const persona = document.getElementById("persona");
const depth = document.getElementById("depth");
const sendBtn = document.getElementById("send");

/* ============================================================
   TEXT FORMAT HELPERS
============================================================ */

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function formatText(text) {
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>")
    .replace(/\*(.*?)\*/g, "<i>$1</i>")
    .replace(/\n/g, "<br>");
}

/* ============================================================
   SAFE SCROLL
   Chỉ cuộn vùng tin nhắn, tuyệt đối không focus input sau khi gửi.
============================================================ */

function scrollToBottom() {
  if (!messages) return;

  requestAnimationFrame(() => {
    messages.scrollTop = messages.scrollHeight;
  });

  setTimeout(() => {
    messages.scrollTop = messages.scrollHeight;
  }, 80);

  setTimeout(() => {
    messages.scrollTop = messages.scrollHeight;
  }, 220);
}

/* ============================================================
   MESSAGE UI
============================================================ */

function addMessage(role, content, meta = "") {
  const row = document.createElement("div");
  row.className = `msg ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role === "user" ? "human" : "ai"}`;
  avatar.textContent = role === "user" ? "YOU" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = formatText(content);

  if (meta) {
    const metaDiv = document.createElement("div");
    metaDiv.className = "meta";
    metaDiv.textContent = meta;
    bubble.appendChild(metaDiv);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  messages.appendChild(row);

  scrollToBottom();

  return bubble;
}

function fillPrompt(text) {
  input.value = text;
  /* Chỉ focus khi người dùng bấm prompt, không dùng sau khi gửi */
  input.focus({ preventScroll: true });
}

/* ============================================================
   SEND MESSAGE
============================================================ */

async function sendMessage() {
  const text = input.value.trim();
  if (!text || sendBtn.disabled) return;

  addMessage("user", text);
  history.push({ role: "user", content: text });

  input.value = "";
  input.style.height = "58px";

  scrollToBottom();

  sendBtn.disabled = true;

  const bubble = addMessage(
    "assistant",
    "Đang phân tích câu hỏi theo ngữ cảnh dự án..."
  );

  const start = performance.now();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message: text,
        history: history,
        persona: persona ? persona.value : "student",
        depth: depth ? depth.value : "balanced"
      })
    });

    const data = await res.json();
    const elapsed = ((performance.now() - start) / 1000).toFixed(1);

    const reply = data.reply || "Mình chưa có câu trả lời phù hợp.";
    const meta = `Intent: ${data.intent || "unknown"} · Source: ${data.source || "ai"} · ${elapsed}s`;

    bubble.innerHTML = "";

    await typeWriter(bubble, reply, meta);

    history.push({
      role: "assistant",
      content: reply
    });

  } catch (err) {
    bubble.innerHTML = formatText(
      "Không kết nối được backend AI. Hãy kiểm tra uvicorn server."
    );

    const metaDiv = document.createElement("div");
    metaDiv.className = "meta";
    metaDiv.textContent = "Backend connection error";
    bubble.appendChild(metaDiv);

    scrollToBottom();

  } finally {
    sendBtn.disabled = false;
    scrollToBottom();
  }
}

/* ============================================================
   TYPEWRITER
============================================================ */

function typeWriter(element, text, meta = "") {
  return new Promise((resolve) => {
    let i = 0;
    element.innerHTML = "";
    scrollToBottom();

    const timer = setInterval(() => {
      element.innerHTML = formatText(text.slice(0, i + 1));
      scrollToBottom();

      i++;

      if (i >= text.length) {
        clearInterval(timer);

        if (meta) {
          const metaDiv = document.createElement("div");
          metaDiv.className = "meta";
          metaDiv.textContent = meta;
          element.appendChild(metaDiv);
        }

        scrollToBottom();
        resolve();
      }
    }, 3);
  });
}

/* ============================================================
   NEW CHAT
============================================================ */

function newChat() {
  history = [];
  messages.innerHTML = "";

  addMessage(
    "assistant",
    "Chat mới đã sẵn sàng 👋 Mình hiểu đây là dự án Credit Risk Nexus. Bạn chọn đối tượng hỏi ở bên trái rồi hỏi tự do nhé."
  );

  input.value = "";
  input.style.height = "58px";

  scrollToBottom();
}

/* ============================================================
   EVENTS
============================================================ */

sendBtn.addEventListener("click", sendMessage);

input.addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

input.addEventListener("input", function () {
  input.style.height = "58px";
});

/* ============================================================
   INIT
============================================================ */

window.addEventListener("load", function () {
  scrollToBottom();
});