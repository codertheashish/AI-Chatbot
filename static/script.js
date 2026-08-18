/* =========================================================================
   AI Chatbot — Frontend logic (plain JS, no frameworks)
   ========================================================================= */

// ---------------------------------------------------------------------------
// Reusable inline icon SVGs (kept as small template strings, no emoji anywhere)
// ---------------------------------------------------------------------------
const ICONS = {
  cross: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
  copy: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
  speaker: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>',
  stop: '<svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>',
  file: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
  user: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
  bot: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><line x1="12" y1="7" x2="12" y2="11"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>',
};

const state = {
  chats: [],
  currentChatId: null,
  documents: [],
  ragEnabled: true,
  voiceOutputEnabled: false,
  isStreaming: false,
};

// ---------------------------------------------------------------------------
// Element references
// ---------------------------------------------------------------------------
const el = {
  sidebar: document.getElementById("sidebar"),
  sidebarBackdrop: document.getElementById("sidebarBackdrop"),
  chatList: document.getElementById("chatList"),
  chatSearch: document.getElementById("chatSearch"),
  newChatBtn: document.getElementById("newChatBtn"),
  sidebarOpenBtn: document.getElementById("sidebarOpenBtn"),
  sidebarCloseBtn: document.getElementById("sidebarCloseBtn"),

  modelSelect: document.getElementById("modelSelect"),
  settingsModelSelect: document.getElementById("settingsModelSelect"),
  topbarTitle: document.getElementById("topbarTitle"),
  renameBtn: document.getElementById("renameBtn"),
  exportBtn: document.getElementById("exportBtn"),
  deleteBtn: document.getElementById("deleteBtn"),
  themeToggle: document.getElementById("themeToggle"),

  chatWindow: document.getElementById("chatWindow"),
  emptyState: document.getElementById("emptyState"),
  messages: document.getElementById("messages"),
  docChipBar: document.getElementById("docChipBar"),

  uploadBtn: document.getElementById("uploadBtn"),
  fileInput: document.getElementById("fileInput"),
  messageInput: document.getElementById("messageInput"),
  micBtn: document.getElementById("micBtn"),
  sendBtn: document.getElementById("sendBtn"),

  settingsBtn: document.getElementById("settingsBtn"),
  settingsModal: document.getElementById("settingsModal"),
  closeSettingsBtn: document.getElementById("closeSettingsBtn"),
  ragToggle: document.getElementById("ragToggle"),
  voiceOutputToggle: document.getElementById("voiceOutputToggle"),

  renameModal: document.getElementById("renameModal"),
  closeRenameBtn: document.getElementById("closeRenameBtn"),
  renameInput: document.getElementById("renameInput"),
  confirmRenameBtn: document.getElementById("confirmRenameBtn"),

  googleSignInBtn: document.getElementById("googleSignInBtn"),
  authProfile: document.getElementById("authProfile"),
  authAvatar: document.getElementById("authAvatar"),
  authName: document.getElementById("authName"),
  signOutBtn: document.getElementById("signOutBtn"),
};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function toast(msg) {
  let t = document.querySelector(".toast");
  if (!t) {
    t = document.createElement("div");
    t.className = "toast";
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 2200);
}

/** Minimal, dependency-free Markdown -> HTML (code blocks, bold/italic, links, lists). */
function renderMarkdown(text) {
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // fenced code blocks ```lang\ncode```
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    return `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`;
  });
  // inline code
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  // bold / italic
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  // links
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  // unordered lists
  html = html.replace(/(?:^|\n)([-*] .+(?:\n[-*] .+)*)/g, (block) => {
    const items = block.trim().split("\n").map(l => `<li>${l.replace(/^[-*]\s+/, "")}</li>`).join("");
    return `<ul>${items}</ul>`;
  });
  // paragraphs (split on blank lines, skip if already block-level)
  html = html
    .split(/\n{2,}/)
    .map(block => (/^<(pre|ul|ol)/.test(block.trim()) ? block : `<p>${block.replace(/\n/g, "<br>")}</p>`))
    .join("");

  return html;
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Sidebar / chat list
// ---------------------------------------------------------------------------

async function loadChats() {
  state.chats = await api("/api/chats");
  renderChatList();
}

function renderChatList(filter = "") {
  el.chatList.innerHTML = "";
  const q = filter.trim().toLowerCase();
  const list = q ? state.chats.filter(c => c.title.toLowerCase().includes(q)) : state.chats;

  for (const chat of list) {
    const item = document.createElement("div");
    item.className = "chat-item" + (chat.id === state.currentChatId ? " active" : "");
    item.innerHTML = `<span class="title"></span><button class="chat-delete" title="Delete">${ICONS.cross}</button>`;
    item.querySelector(".title").textContent = chat.title || "New Chat";
    item.addEventListener("click", (e) => {
      if (e.target.closest(".chat-delete")) return;
      openChat(chat.id);
      closeSidebarOnMobile();
    });
    item.querySelector(".chat-delete").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm("Delete this chat?")) return;
      await api(`/api/chats/${chat.id}`, { method: "DELETE" });
      if (state.currentChatId === chat.id) {
        state.currentChatId = null;
        el.messages.innerHTML = "";
        el.emptyState.classList.remove("hidden");
      }
      await loadChats();
    });
    el.chatList.appendChild(item);
  }
}

el.chatSearch.addEventListener("input", () => renderChatList(el.chatSearch.value));

el.newChatBtn.addEventListener("click", () => {
  // Just switch to a blank composer -- no DB row is created until the user
  // actually sends a message (see ensureChat()). Old chats stay untouched in history.
  showBlankChatView();
  closeSidebarOnMobile();
});

function showBlankChatView() {
  state.currentChatId = null;
  state.documents = [];
  el.messages.innerHTML = "";
  el.emptyState.classList.remove("hidden");
  el.topbarTitle.textContent = "New chat";
  renderDocChips();
  renderChatList(el.chatSearch.value);
}

// ---------------------------------------------------------------------------
// Open / render a chat
// ---------------------------------------------------------------------------

async function openChat(chatId) {
  state.currentChatId = chatId;
  const data = await api(`/api/chats/${chatId}`);
  el.modelSelect.value = data.chat.model;
  el.topbarTitle.textContent = data.chat.title || "New chat";
  el.emptyState.classList.add("hidden");
  el.messages.innerHTML = "";
  data.messages.forEach(m => appendMessage(m.role, m.content, m.sources || [], false));
  state.documents = data.documents || [];
  renderDocChips();
  renderChatList(el.chatSearch.value);
  scrollToBottom();
}

function renderDocChips() {
  el.docChipBar.innerHTML = "";
  for (const doc of state.documents) {
    const chip = document.createElement("div");
    chip.className = "doc-chip";
    chip.innerHTML = `<span class="doc-chip-label">${ICONS.file}${escapeHtml(doc.filename)}</span><button title="Remove">${ICONS.cross}</button>`;
    chip.querySelector("button").addEventListener("click", async () => {
      await api(`/api/documents/${doc.id}`, { method: "DELETE" });
      state.documents = state.documents.filter(d => d.id !== doc.id);
      renderDocChips();
      toast("Document removed");
    });
    el.docChipBar.appendChild(chip);
  }
}

function scrollToBottom() {
  el.chatWindow.scrollTop = el.chatWindow.scrollHeight;
}

function appendMessage(role, content, sources = [], animate = true) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  msg.innerHTML = `
    <div class="msg-avatar">${role === "user" ? ICONS.user : ICONS.bot}</div>
    <div class="msg-body">
      <div class="msg-role">${role === "user" ? "You" : "Assistant"}</div>
      <div class="msg-content"></div>
      <div class="msg-sources"></div>
      <div class="msg-actions">
        <button class="copy-btn" title="Copy">${ICONS.copy}</button>
        ${role === "assistant" ? `<button class="speak-btn" title="Read aloud">${ICONS.speaker}</button>` : ""}
      </div>
    </div>
  `;
  const contentEl = msg.querySelector(".msg-content");
  contentEl.innerHTML = renderMarkdown(content);

  if (sources.length) {
    const wrap = msg.querySelector(".msg-sources");
    sources.forEach(s => {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.innerHTML = `${ICONS.file}<span>${escapeHtml(s.filename)}</span>`;
      wrap.appendChild(chip);
    });
  }

  msg.querySelector(".copy-btn").addEventListener("click", () => {
    navigator.clipboard.writeText(contentEl.innerText);
    toast("Copied to clipboard");
  });
  const speakBtn = msg.querySelector(".speak-btn");
  if (speakBtn) speakBtn.addEventListener("click", () => toggleSpeak(contentEl.innerText, speakBtn));

  el.messages.appendChild(msg);
  if (animate) scrollToBottom();
  return contentEl;
}

// ---------------------------------------------------------------------------
// Sending messages + SSE streaming
// ---------------------------------------------------------------------------

function autoResizeTextarea() {
  el.messageInput.style.height = "auto";
  el.messageInput.style.height = Math.min(el.messageInput.scrollHeight, 200) + "px";
}
el.messageInput.addEventListener("input", autoResizeTextarea);
el.messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
el.sendBtn.addEventListener("click", sendMessage);

async function ensureChat() {
  if (state.currentChatId) return state.currentChatId;
  const chat = await api("/api/chats", {
    method: "POST",
    body: JSON.stringify({ title: "New Chat", model: el.modelSelect.value }),
  });
  await loadChats();
  state.currentChatId = chat.id;
  el.emptyState.classList.add("hidden");
  return chat.id;
}

async function sendMessage() {
  const text = el.messageInput.value.trim();
  if (!text || state.isStreaming) return;

  const chatId = await ensureChat();
  el.messageInput.value = "";
  autoResizeTextarea();

  appendMessage("user", text, [], true);

  // Placeholder assistant bubble with typing indicator while streaming
  const contentEl = appendMessage("assistant", "", [], true);
  contentEl.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';

  state.isStreaming = true;
  el.sendBtn.disabled = true;

  try {
    const resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: chatId,
        message: text,
        model: el.modelSelect.value,
        use_rag: state.ragEnabled,
      }),
    });

    if (!resp.ok || !resp.body) throw new Error(`Server error: ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";
    let sources = [];
    let firstToken = true;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // keep incomplete chunk in buffer

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const jsonStr = line.slice(5).trim();
        let evt;
        try { evt = JSON.parse(jsonStr); } catch { continue; }

        if (evt.type === "sources") {
          sources = evt.sources || [];
        } else if (evt.type === "token") {
          if (firstToken) { contentEl.innerHTML = ""; firstToken = false; }
          fullText += evt.content;
          contentEl.innerHTML = renderMarkdown(fullText);
          scrollToBottom();
        } else if (evt.type === "error") {
          contentEl.innerHTML = `
            <div class="error-block">
              <p class="error-title">${escapeHtml(evt.title)}</p>
              <p class="error-solution">${escapeHtml(evt.solution)}</p>
            </div>`;
          scrollToBottom();
        } else if (evt.type === "done") {
          if (sources.length) {
            const wrap = contentEl.parentElement.querySelector(".msg-sources");
            sources.forEach(s => {
              const chip = document.createElement("span");
              chip.className = "source-chip";
              chip.innerHTML = `${ICONS.file}<span>${escapeHtml(s.filename)}</span>`;
              wrap.appendChild(chip);
            });
          }
          if (state.voiceOutputEnabled) speak(fullText);
        }
      }
    }
    await loadChats(); // refresh sidebar (title / ordering may have changed)
  } catch (err) {
    contentEl.innerHTML = `
      <div class="error-block">
        <p class="error-title">Couldn't get a response.</p>
        <p class="error-solution">${escapeHtml(err.message)}</p>
      </div>`;
  } finally {
    state.isStreaming = false;
    el.sendBtn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// File upload (RAG ingestion)
// ---------------------------------------------------------------------------

el.uploadBtn.addEventListener("click", () => el.fileInput.click());

el.fileInput.addEventListener("change", async () => {
  const file = el.fileInput.files[0];
  if (!file) return;
  const chatId = await ensureChat();

  const formData = new FormData();
  formData.append("chat_id", chatId);
  formData.append("file", file);

  toast(`Uploading ${file.name}...`);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    state.documents.push({ id: data.id, filename: data.filename });
    renderDocChips();
    toast(`${data.filename} indexed (${data.chunks} chunks)`);
  } catch (err) {
    toast("Upload failed: " + err.message);
  } finally {
    el.fileInput.value = "";
  }
});

// ---------------------------------------------------------------------------
// Voice input (Web Speech API) + Voice output (SpeechSynthesis)
// ---------------------------------------------------------------------------

let recognizer = null;
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
  recognizer = new SpeechRecognition();
  recognizer.continuous = false;
  recognizer.interimResults = true;
  recognizer.lang = "en-US";

  recognizer.onresult = (event) => {
    let transcript = "";
    for (let i = 0; i < event.results.length; i++) transcript += event.results[i][0].transcript;
    el.messageInput.value = transcript;
    autoResizeTextarea();
  };
  recognizer.onend = () => el.micBtn.classList.remove("recording");
  recognizer.onerror = () => { el.micBtn.classList.remove("recording"); toast("Voice input error"); };
} else {
  el.micBtn.title = "Voice input not supported in this browser";
}

el.micBtn.addEventListener("click", () => {
  if (!recognizer) { toast("Voice input isn't supported in this browser"); return; }
  if (el.micBtn.classList.contains("recording")) {
    recognizer.stop();
  } else {
    recognizer.start();
    el.micBtn.classList.add("recording");
  }
});

let activeSpeakBtn = null; // tracks which "Speak" button (if any) is currently reading aloud

/** Toggle behavior: click once to speak, click the SAME button again to stop. */
function toggleSpeak(text, btn) {
  if (!("speechSynthesis" in window)) { toast("Voice output not supported"); return; }

  const isThisButtonSpeaking = activeSpeakBtn === btn && window.speechSynthesis.speaking;

  // Always stop whatever is currently playing first (only one utterance at a time)
  window.speechSynthesis.cancel();
  if (activeSpeakBtn) resetSpeakBtn(activeSpeakBtn);

  if (isThisButtonSpeaking) {
    // User clicked the same button again -> just stop, don't restart
    activeSpeakBtn = null;
    return;
  }

  const utter = new SpeechSynthesisUtterance(text.slice(0, 4000));
  utter.rate = 1.0;
  utter.onend = () => { resetSpeakBtn(btn); activeSpeakBtn = null; };
  utter.onerror = () => { resetSpeakBtn(btn); activeSpeakBtn = null; };

  activeSpeakBtn = btn;
  btn.innerHTML = ICONS.stop;
  btn.title = "Stop";
  btn.classList.add("speaking");
  window.speechSynthesis.speak(utter);
}

function resetSpeakBtn(btn) {
  btn.innerHTML = ICONS.speaker;
  btn.title = "Read aloud";
  btn.classList.remove("speaking");
}

/** Kept for programmatic calls (e.g. auto-read after streaming) — always starts fresh. */
function speak(text) {
  if (!("speechSynthesis" in window)) { toast("Voice output not supported"); return; }
  window.speechSynthesis.cancel();
  if (activeSpeakBtn) { resetSpeakBtn(activeSpeakBtn); activeSpeakBtn = null; }
  const utter = new SpeechSynthesisUtterance(text.slice(0, 4000));
  utter.rate = 1.0;
  window.speechSynthesis.speak(utter);
}

// ---------------------------------------------------------------------------
// Rename / Delete / Export current chat
// ---------------------------------------------------------------------------

el.renameBtn.addEventListener("click", () => {
  if (!state.currentChatId) return toast("Open a chat first");
  const current = state.chats.find(c => c.id === state.currentChatId);
  el.renameInput.value = current ? current.title : "";
  el.renameModal.classList.add("open");
  el.renameInput.focus();
});
el.closeRenameBtn.addEventListener("click", () => el.renameModal.classList.remove("open"));
el.confirmRenameBtn.addEventListener("click", async () => {
  const title = el.renameInput.value.trim();
  if (!title || !state.currentChatId) return;
  await api(`/api/chats/${state.currentChatId}`, { method: "PUT", body: JSON.stringify({ title }) });
  el.renameModal.classList.remove("open");
  el.topbarTitle.textContent = title;
  await loadChats();
});

el.deleteBtn.addEventListener("click", async () => {
  if (!state.currentChatId) return toast("Open a chat first");
  if (!confirm("Delete this chat permanently?")) return;
  await api(`/api/chats/${state.currentChatId}`, { method: "DELETE" });
  state.currentChatId = null;
  el.messages.innerHTML = "";
  el.emptyState.classList.remove("hidden");
  await loadChats();
});

el.exportBtn.addEventListener("click", () => {
  if (!state.currentChatId) return toast("Open a chat first");
  window.open(`/api/chats/${state.currentChatId}/export?fmt=md`, "_blank");
});

// ---------------------------------------------------------------------------
// Settings modal
// ---------------------------------------------------------------------------

el.settingsBtn.addEventListener("click", () => {
  el.settingsModelSelect.value = el.modelSelect.value;
  el.settingsModal.classList.add("open");
});
el.closeSettingsBtn.addEventListener("click", () => el.settingsModal.classList.remove("open"));

el.settingsModelSelect.addEventListener("change", () => {
  el.modelSelect.value = el.settingsModelSelect.value;
});
el.ragToggle.addEventListener("change", () => { state.ragEnabled = el.ragToggle.checked; });
el.voiceOutputToggle.addEventListener("change", () => { state.voiceOutputEnabled = el.voiceOutputToggle.checked; });

document.querySelectorAll(".theme-btn").forEach(btn => {
  btn.addEventListener("click", () => setTheme(btn.dataset.theme));
});

// ---------------------------------------------------------------------------
// Theme (dark mode default, persisted in localStorage)
// ---------------------------------------------------------------------------

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
  const moonIcon = document.getElementById("themeIconMoon");
  const sunIcon = document.getElementById("themeIconSun");
  if (moonIcon && sunIcon) {
    moonIcon.style.display = theme === "dark" ? "block" : "none";
    sunIcon.style.display = theme === "dark" ? "none" : "block";
  }
}
el.themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  setTheme(current === "dark" ? "light" : "dark");
});
setTheme(localStorage.getItem("theme") || "dark");

// ---------------------------------------------------------------------------
// Mobile sidebar toggle
// ---------------------------------------------------------------------------

function openSidebar() {
  el.sidebar.classList.add("open");
  el.sidebarBackdrop.classList.add("open");
}
function closeSidebar() {
  el.sidebar.classList.remove("open");
  el.sidebarBackdrop.classList.remove("open");
}
function closeSidebarOnMobile() {
  if (window.innerWidth <= 768) closeSidebar();
}
el.sidebarOpenBtn.addEventListener("click", openSidebar);
el.sidebarCloseBtn.addEventListener("click", closeSidebar);
el.sidebarBackdrop.addEventListener("click", closeSidebar);

// ---------------------------------------------------------------------------
// Google sign-in (client-side only, via Google Identity Services)
// ---------------------------------------------------------------------------
// To enable: create an OAuth Client ID at https://console.cloud.google.com/apis/credentials
// (type "Web application", add your app's URL under "Authorized JavaScript origins"),
// then paste the Client ID below. Leave blank to keep the button disabled.
const GOOGLE_CLIENT_ID = "601637157120-sgvrp0j4loou45g17n3ptj93lm1obm9b.apps.googleusercontent.com";

function initGoogleSignIn() {
  const saved = localStorage.getItem("authUser");
  if (saved) {
    try { showSignedInUser(JSON.parse(saved)); } catch { /* ignore corrupt value */ }
  }

  if (!GOOGLE_CLIENT_ID) {
    el.googleSignInBtn.addEventListener("click", () => {
      toast("Sign-in isn't set up yet — add a Google Client ID in script.js to enable it");
    });
    return;
  }

  const trySetup = () => {
    if (!window.google || !window.google.accounts) { setTimeout(trySetup, 300); return; }
    window.google.accounts.id.initialize({
      client_id: GOOGLE_CLIENT_ID,
      callback: handleGoogleCredential,
    });
    el.googleSignInBtn.addEventListener("click", () => window.google.accounts.id.prompt());
  };
  trySetup();
}

function handleGoogleCredential(response) {
  try {
    // Decode the JWT payload (base64url) to read basic profile info.
    const payload = JSON.parse(atob(response.credential.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
    const user = { name: payload.name, email: payload.email, picture: payload.picture };
    localStorage.setItem("authUser", JSON.stringify(user));
    showSignedInUser(user);
    toast(`Signed in as ${user.name}`);
  } catch {
    toast("Sign-in failed — please try again");
  }
}

function showSignedInUser(user) {
  el.googleSignInBtn.style.display = "none";
  el.authProfile.style.display = "flex";
  el.authAvatar.src = user.picture || "";
  el.authName.textContent = user.name || user.email || "Signed in";
}

el.signOutBtn.addEventListener("click", () => {
  localStorage.removeItem("authUser");
  el.authProfile.style.display = "none";
  el.googleSignInBtn.style.display = "flex";
  if (window.google && window.google.accounts) window.google.accounts.id.disableAutoSelect();
  toast("Signed out");
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

(async function init() {
  // Explicitly force the model dropdown to the server's configured default.
  // (Browsers sometimes restore a <select>'s previous value on reload/back-
  // navigation, which made an old manually-picked model "stick" instead of
  // the one set in .env -- this line guarantees the configured default wins.)
  const defaultModel = document.body.dataset.defaultModel;
  if (defaultModel) {
    el.modelSelect.value = defaultModel;
    el.settingsModelSelect.value = defaultModel;
  }

  try {
    await loadChats();
    // Always start on a blank "new chat" view -- old chats remain in the
    // sidebar for the user to reopen, but are never auto-opened on load.
    showBlankChatView();
  } catch (err) {
    toast("Failed to load chats: " + err.message);
  }
  initGoogleSignIn();
})();
