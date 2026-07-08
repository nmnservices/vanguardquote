(function () {
  const API = 'https://web-production-0d92f.up.railway.app/chat';

  // ── Styles ──────────────────────────────────────────────────
  const style = document.createElement('style');
  style.textContent = `
    #vq-trigger {
      position: fixed;
      bottom: 24px;
      right: 24px;
      width: 56px;
      height: 56px;
      background: #16a34a;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(22,163,74,0.4);
      z-index: 999999;
      transition: transform 0.2s;
      border: none;
    }
    #vq-trigger:hover { transform: scale(1.05); }
    #vq-trigger svg { width: 24px; height: 24px; fill: white; }

    #vq-window {
      position: fixed;
      bottom: 90px;
      right: 24px;
      width: 360px;
      height: 540px;
      background: white;
      border-radius: 16px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.15);
      display: none;
      flex-direction: column;
      overflow: hidden;
      z-index: 999998;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    #vq-window.vq-open { display: flex; }

    #vq-header {
      background: #16a34a;
      color: white;
      padding: 16px 20px;
      display: flex;
      align-items: center;
      gap: 12px;
      flex-shrink: 0;
    }
    .vq-avatar {
      width: 36px;
      height: 36px;
      background: rgba(255,255,255,0.2);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      flex-shrink: 0;
    }
    .vq-header-info .vq-name { font-weight: 600; font-size: 15px; }
    .vq-header-info .vq-status { font-size: 12px; opacity: 0.85; }

    #vq-messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .vq-msg {
      max-width: 82%;
      padding: 10px 14px;
      border-radius: 16px;
      font-size: 14px;
      line-height: 1.5;
    }
    .vq-msg.vq-bot {
      background: #f1f5f9;
      color: #1e293b;
      align-self: flex-start;
      border-bottom-left-radius: 4px;
    }
    .vq-msg.vq-user {
      background: #16a34a;
      color: white;
      align-self: flex-end;
      border-bottom-right-radius: 4px;
    }
    .vq-msg.vq-typing {
      background: #f1f5f9;
      color: #94a3b8;
      align-self: flex-start;
      font-style: italic;
      font-size: 13px;
    }

    .vq-quote-card {
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      border-radius: 12px;
      padding: 14px;
      font-size: 13px;
      align-self: flex-start;
      max-width: 85%;
    }
    .vq-quote-card .vq-price {
      font-size: 20px;
      font-weight: 700;
      color: #16a34a;
      margin: 4px 0 8px;
    }
    .vq-quote-card .vq-qlabel { color: #64748b; font-size: 12px; }
    .vq-quote-card .vq-service { font-weight: 600; color: #1e293b; }

    #vq-prechat {
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      flex: 1;
    }
    #vq-prechat p {
      font-size: 14px;
      color: #475569;
      margin: 0 0 4px;
    }
    .vq-input-field {
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 14px;
      outline: none;
      width: 100%;
      box-sizing: border-box;
      font-family: inherit;
    }
    .vq-input-field:focus { border-color: #16a34a; }
    .vq-start-btn {
      background: #16a34a;
      color: white;
      border: none;
      border-radius: 8px;
      padding: 12px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      width: 100%;
      font-family: inherit;
    }
    .vq-start-btn:hover { background: #15803d; }

    #vq-input-area {
      padding: 12px 16px;
      border-top: 1px solid #e2e8f0;
      display: flex;
      gap: 8px;
      flex-shrink: 0;
    }
    #vq-input {
      flex: 1;
      border: 1px solid #e2e8f0;
      border-radius: 24px;
      padding: 10px 16px;
      font-size: 14px;
      outline: none;
      resize: none;
      font-family: inherit;
      max-height: 80px;
    }
    #vq-input:focus { border-color: #16a34a; }
    #vq-send {
      width: 40px;
      height: 40px;
      background: #16a34a;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    #vq-send:hover { background: #15803d; }
    #vq-send svg { width: 18px; height: 18px; fill: white; }

    @media (max-width: 420px) {
      #vq-window {
        width: calc(100vw - 24px);
        right: 12px;
        bottom: 80px;
      }
    }
  `;
  document.head.appendChild(style);

  // ── State ────────────────────────────────────────────────────
  let sessionId = null;
  let isOpen = false;
  let prechatDone = false;
  let leadName = '';
  let leadPhone = '';
  let quoteShown = false;

  // ── DOM ──────────────────────────────────────────────────────
  const trigger = document.createElement('button');
  trigger.id = 'vq-trigger';
  trigger.setAttribute('aria-label', 'Get an instant quote');
  trigger.innerHTML = `<svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg>`;

  const win = document.createElement('div');
  win.id = 'vq-window';
  win.innerHTML = `
    <div id="vq-header">
      <div class="vq-avatar">🏠</div>
      <div class="vq-header-info">
        <div class="vq-name">Vanguard Haul</div>
        <div class="vq-status">● Get an instant quote</div>
      </div>
    </div>
    <div id="vq-prechat">
      <p>Enter your details to get an instant quote for your job.</p>
      <input class="vq-input-field" id="vq-name-input" type="text" placeholder="Your name" />
      <input class="vq-input-field" id="vq-phone-input" type="tel" placeholder="Phone number" />
      <button class="vq-start-btn" id="vq-start-btn">Get My Quote →</button>
    </div>
    <div id="vq-messages" style="display:none"></div>
    <div id="vq-input-area" style="display:none">
      <textarea id="vq-input" rows="1" placeholder="Describe your job..."></textarea>
      <button id="vq-send">
        <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
      </button>
    </div>
  `;

  document.body.appendChild(trigger);
  document.body.appendChild(win);

  // ── Helpers ──────────────────────────────────────────────────
  function toggleChat() {
    isOpen = !isOpen;
    win.classList.toggle('vq-open', isOpen);
  }

  function addMsg(role, text) {
    const msgs = document.getElementById('vq-messages');
    const div = document.createElement('div');
    div.className = `vq-msg vq-${role}`;
    div.innerHTML = text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>');
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div;
  }

  function addQuoteCard(service, min, max, currency) {
    const msgs = document.getElementById('vq-messages');
    const card = document.createElement('div');
    card.className = 'vq-quote-card';
    card.innerHTML = `
      <div class="vq-qlabel">Instant quote for</div>
      <div class="vq-service">${service.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
      <div class="vq-price">$${min.toLocaleString()} – $${max.toLocaleString()} ${currency}</div>
      <div class="vq-qlabel">✅ No obligation · Free to book</div>
    `;
    msgs.appendChild(card);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function showTyping() {
    return addMsg('typing', 'typing...');
  }

  function switchToChat() {
    document.getElementById('vq-prechat').style.display = 'none';
    document.getElementById('vq-messages').style.display = 'flex';
    document.getElementById('vq-input-area').style.display = 'flex';
    prechatDone = true;
  }

  // ── Pre-chat form submit ──────────────────────────────────────
  document.getElementById('vq-start-btn').addEventListener('click', async function () {
    const nameInput = document.getElementById('vq-name-input').value.trim();
    const phoneInput = document.getElementById('vq-phone-input').value.trim();

    if (!nameInput || !phoneInput) {
      alert('Please enter your name and phone number to continue.');
      return;
    }

    leadName = nameInput;
    leadPhone = phoneInput;
    switchToChat();

    // Welcome message — ask for job description before calling API
    addMsg('bot', `Hi ${leadName}! 👋 Thanks for reaching out to Vanguard Haul.\n\nWhat can we help you with today? Describe your job and location and we'll get you an instant quote.`);
  });

  // ── Send message ─────────────────────────────────────────────
  async function sendMessage() {
    const input = document.getElementById('vq-input');
    const text = input.value.trim();
    if (!text) return;
    addMsg('user', text);
    input.value = '';
    input.style.height = 'auto';

    // On first message, prepend name and phone so agent has full context
    const isFirstMessage = sessionId === null;
    const msgToSend = isFirstMessage
      ? `My name is ${leadName} and my phone number is ${leadPhone}. ${text}`
      : text;

    await sendToAPI(msgToSend);
  }

  async function sendToAPI(text) {
    const typing = showTyping();
    try {
      const res = await fetch(API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: text })
      });
      const data = await res.json();
      sessionId = data.session_id;
      typing.remove();
      addMsg('bot', data.reply);

      // Show quote card once only
      if (data.price_range && data.price_range.min && !quoteShown) {
        addQuoteCard(data.service_type, data.price_range.min, data.price_range.max, data.price_range.currency);
        quoteShown = true;
      }
    } catch (err) {
      typing.remove();
      addMsg('bot', 'Sorry, something went wrong. Please call us at 647-874-2996.');
    }
  }

  // ── Event listeners ──────────────────────────────────────────
  trigger.addEventListener('click', toggleChat);

  document.getElementById('vq-send').addEventListener('click', sendMessage);

  document.getElementById('vq-input').addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  document.getElementById('vq-input').addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 80) + 'px';
  });

})();