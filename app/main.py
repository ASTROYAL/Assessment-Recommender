from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

try:
    from . import agent, retriever
except ImportError:
    import agent
    import retriever


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_must_be_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("content must be a non-empty string")
        return cleaned


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)

    @model_validator(mode="after")
    def last_message_must_be_user(self) -> "ChatRequest":
        if self.messages[-1].role != "user":
            raise ValueError("last message must have role user")
        return self


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


APP_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SHL Assessment Recommender</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #65717f;
      --line: #d8dee6;
      --accent: #0067a6;
      --accent-strong: #004f80;
      --user: #e8f2fb;
      --assistant: #ffffff;
      --shadow: 0 18px 45px rgba(23, 32, 42, 0.12);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }

    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      max-width: 1040px;
      margin: 0 auto;
      padding: 24px;
      gap: 16px;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 14px 0;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .mark {
      width: 38px;
      height: 38px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: var(--accent);
      color: #ffffff;
      font-weight: 700;
      flex: 0 0 auto;
    }

    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      font-weight: 700;
    }

    .status {
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
    }

    .chat {
      min-height: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      display: grid;
      grid-template-rows: 1fr auto;
      overflow: hidden;
    }

    .messages {
      min-height: 420px;
      max-height: calc(100vh - 210px);
      overflow-y: auto;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .empty {
      margin: auto;
      width: min(560px, 100%);
      text-align: center;
      color: var(--muted);
      line-height: 1.5;
      font-size: 16px;
    }

    .message {
      width: min(760px, 92%);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 15px;
      line-height: 1.45;
      font-size: 15px;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }

    .message.user {
      align-self: flex-end;
      background: var(--user);
      border-color: #c4dcee;
    }

    .message.assistant {
      align-self: flex-start;
      background: var(--assistant);
    }

    .recommendations {
      width: min(820px, 96%);
      align-self: flex-start;
      display: grid;
      gap: 10px;
      margin-top: -4px;
    }

    .recommendation {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 15px;
      background: #fbfcfd;
      display: grid;
      gap: 8px;
    }

    .recommendation-top {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }

    .recommendation a {
      color: var(--accent-strong);
      font-weight: 700;
      text-decoration: none;
    }

    .recommendation a:hover {
      text-decoration: underline;
    }

    .type {
      flex: 0 0 auto;
      min-width: 34px;
      height: 28px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: #ffffff;
      background: #44515f;
      font-weight: 700;
      font-size: 13px;
    }

    form {
      border-top: 1px solid var(--line);
      padding: 16px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      background: #ffffff;
    }

    textarea {
      width: 100%;
      min-height: 50px;
      max-height: 150px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 14px;
      font: inherit;
      line-height: 1.35;
      color: var(--ink);
      outline: none;
    }

    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(0, 103, 166, 0.16);
    }

    button {
      border: 0;
      border-radius: 8px;
      padding: 0 20px;
      min-width: 92px;
      font: inherit;
      font-weight: 700;
      color: #ffffff;
      background: var(--accent);
      cursor: pointer;
    }

    button:hover {
      background: var(--accent-strong);
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.62;
    }

    .error {
      color: #9b1c31;
    }

    @media (max-width: 700px) {
      .shell {
        padding: 14px;
      }

      header {
        align-items: flex-start;
        flex-direction: column;
      }

      .messages {
        min-height: 360px;
        max-height: calc(100vh - 230px);
        padding: 14px;
      }

      .message,
      .recommendations {
        width: 100%;
      }

      form {
        grid-template-columns: 1fr;
      }

      button {
        min-height: 44px;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div class="brand">
        <div class="mark">SHL</div>
        <h1>Assessment Recommender</h1>
      </div>
      <div class="status" id="status">Ready</div>
    </header>

    <section class="chat" aria-label="Chat">
      <div class="messages" id="messages">
        <div class="empty" id="empty">Tell me the role, seniority, and key skills you want to assess.</div>
      </div>

      <form id="chat-form">
        <textarea id="input" name="message" rows="2" placeholder="Example: Mid-level data analyst with SQL, Excel, and numerical reasoning" required></textarea>
        <button id="send" type="submit">Send</button>
      </form>
    </section>
  </main>

  <script>
    const form = document.getElementById("chat-form");
    const input = document.getElementById("input");
    const send = document.getElementById("send");
    const messagesEl = document.getElementById("messages");
    const emptyEl = document.getElementById("empty");
    const statusEl = document.getElementById("status");
    const history = [];

    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.className = isError ? "status error" : "status";
    }

    function appendMessage(role, content) {
      if (emptyEl) {
        emptyEl.remove();
      }
      const message = document.createElement("div");
      message.className = `message ${role}`;
      message.textContent = content;
      messagesEl.appendChild(message);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function appendRecommendations(items) {
      if (!items || items.length === 0) {
        return;
      }

      const wrap = document.createElement("div");
      wrap.className = "recommendations";
      for (const item of items) {
        const card = document.createElement("article");
        card.className = "recommendation";

        const top = document.createElement("div");
        top.className = "recommendation-top";

        const link = document.createElement("a");
        link.href = item.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = item.name;

        const type = document.createElement("div");
        type.className = "type";
        type.textContent = item.test_type;

        top.appendChild(link);
        top.appendChild(type);
        card.appendChild(top);
        wrap.appendChild(card);
      }
      messagesEl.appendChild(wrap);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    async function sendMessage(text) {
      history.push({ role: "user", content: text });
      appendMessage("user", text);

      if (history.length > 8) {
        const resetText = "The conversation limit was reached. Start a new shortlist with your latest request.";
        history.splice(0, history.length, { role: "user", content: text });
        appendMessage("assistant", resetText);
      }

      setStatus("Thinking");
      send.disabled = true;
      input.disabled = true;

      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: history })
        });

        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          throw new Error(detail.detail || "Request failed");
        }

        const data = await response.json();
        history.push({ role: "assistant", content: data.reply });
        appendMessage("assistant", data.reply);
        appendRecommendations(data.recommendations);
        setStatus(data.end_of_conversation ? "Complete" : "Ready");
      } catch (error) {
        appendMessage("assistant", "I could not get a response. Please try again.");
        setStatus(error.message || "Error", true);
      } finally {
        send.disabled = false;
        input.disabled = false;
        input.focus();
      }
    }

    form.addEventListener("submit", event => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text) {
        return;
      }
      input.value = "";
      sendMessage(text);
    });
  </script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        retriever.load_index()
        app.state.retriever_ready = True
        app.state.retriever_error = None
        print("Retriever index loaded successfully.")
    except Exception as exc:
        app.state.retriever_ready = False
        app.state.retriever_error = str(exc)
        print(f"Retriever startup error: {exc}")
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": "Invalid request body."})


@app.get("/", response_class=HTMLResponse)
async def chat_ui() -> HTMLResponse:
    return HTMLResponse(APP_HTML)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if len(request.messages) > 8:
        raise HTTPException(status_code=400, detail="Maximum 8 messages total.")

    message_payload = [message.model_dump() for message in request.messages]
    try:
        agent_response = await asyncio.wait_for(agent.get_reply(message_payload), timeout=25.0)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Agent timed out. Please try again.") from exc
    except Exception as exc:
        print(f"Unhandled chat error: {exc}")
        agent_response = dict(agent.SAFE_FALLBACK)

    try:
        return ChatResponse.model_validate(agent_response)
    except ValidationError as exc:
        print(f"Invalid agent response schema: {exc}")
        return ChatResponse.model_validate(agent.SAFE_FALLBACK)
