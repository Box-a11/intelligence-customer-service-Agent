const INTENT_LABELS = {
  qa: '问答',
  consultation: '咨询',
  complaint: '投诉',
  unclear: '模糊',
};

let sessionId = 'web-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 7);

// 用户级长期记忆标识：固定在同一浏览器，跨会话（新会话）仍保持记忆
let userId = localStorage.getItem('cs_user_id');
if (!userId) {
  userId = 'user-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 7);
  localStorage.setItem('cs_user_id', userId);
}

const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const sessionIdEl = document.getElementById('session-id');
const backendBadge = document.getElementById('backend-badge');
const ticketsEl = document.getElementById('tickets');
const sessionsEl = document.getElementById('sessions');

function setSessionId(id) {
  sessionId = id;
  sessionIdEl.textContent = id;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function timeAgo(iso) {
  if (!iso) return '';
  const t = new Date(iso);
  if (isNaN(t)) return '';
  const diff = (Date.now() - t.getTime()) / 1000;
  if (diff < 60) return '刚刚';
  if (diff < 3600) return Math.floor(diff / 60) + ' 分钟前';
  if (diff < 86400) return Math.floor(diff / 3600) + ' 小时前';
  return Math.floor(diff / 86400) + ' 天前';
}

function addMessage(role, text, opts = {}) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);

  const meta = document.createElement('div');
  meta.className = 'meta';

  if (role === 'assistant' && !opts.error) {
    if (opts.intent) {
      const b = document.createElement('span');
      b.className = 'badge badge-intent';
      b.textContent = INTENT_LABELS[opts.intent] || opts.intent;
      meta.appendChild(b);
    }
    if (opts.needsClarification) {
      const b = document.createElement('span');
      b.className = 'badge badge-gray';
      b.textContent = '待澄清';
      meta.appendChild(b);
    }
    if (opts.ticket) {
      const b = document.createElement('span');
      b.className = 'badge badge-ticket';
      b.textContent = '已转人工 · ' + opts.ticket.id;
      meta.appendChild(b);
    }
    (opts.sources || []).forEach((s) => {
      const c = document.createElement('span');
      c.className = 'src-chip';
      c.textContent = '📄 ' + s;
      meta.appendChild(c);
    });
  } else if (opts.error) {
    const b = document.createElement('span');
    b.className = 'badge badge-red';
    b.textContent = '出错';
    meta.appendChild(b);
  }

  wrap.appendChild(meta);
  messagesEl.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function addTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'msg assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || sendBtn.disabled) return;
  inputEl.value = '';
  inputEl.style.height = 'auto';
  addMessage('user', text);
  const typing = addTyping();
  sendBtn.disabled = true;
  sendBtn.textContent = '思考中…';
  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: sessionId, user_id: userId }),
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    typing.remove();
    addMessage('assistant', data.reply || '（空回复）', {
      intent: data.intent,
      needsClarification: data.needs_clarification,
      ticket: data.ticket,
      sources: data.sources,
    });
    loadTickets();
    loadSessions();
  } catch (e) {
    typing.remove();
    addMessage('assistant', '请求失败：' + e.message, { error: true });
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = '发送';
    inputEl.focus();
  }
}

function newSession() {
  setSessionId('web-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 7));
  messagesEl.innerHTML = '';
  addMessage('assistant', '你好！我是优选商城智能客服，请问有什么可以帮您？', { intent: null });
  loadSessions();
}

async function loadSessions() {
  try {
    const list = await (await fetch('/sessions')).json();
    sessionsEl.innerHTML = '';
    if (!list.length) {
      sessionsEl.innerHTML = '<div class="empty">暂无历史对话</div>';
      return;
    }
    list.forEach((s) => {
      const item = document.createElement('div');
      item.className = 'session-item' + (s.session_id === sessionId ? ' active' : '');
      item.innerHTML =
        '<div class="session-title">' + escapeHtml(s.preview || '（空会话）') + '</div>' +
        '<div class="session-meta">' + s.message_count + ' 条消息 · ' + timeAgo(s.updated_at) + '</div>';
      item.addEventListener('click', () => viewSession(s.session_id));
      sessionsEl.appendChild(item);
    });
  } catch (e) {
    sessionsEl.innerHTML = '<div class="empty">历史加载失败</div>';
  }
}

async function viewSession(id) {
  setSessionId(id);
  try {
    const data = await (await fetch('/sessions/' + id)).json();
    messagesEl.innerHTML = '';
    (data.messages || []).forEach((m) => {
      if (m.role === 'user') addMessage('user', m.content);
      else addMessage('assistant', m.content, { intent: null });
    });
    loadSessions(); // 刷新高亮
  } catch (e) {
    addMessage('assistant', '加载历史失败：' + e.message, { error: true });
  }
}

function toggleSidebar() {
  document.body.classList.toggle('history-collapsed');
}

async function loadTickets() {
  try {
    const tickets = await (await fetch('/tickets')).json();
    if (!tickets.length) {
      ticketsEl.innerHTML = '<div class="empty">暂无工单</div>';
      return;
    }
    ticketsEl.innerHTML = '';
    tickets.slice().reverse().forEach((t) => {
      const el = document.createElement('div');
      el.className = 'ticket';
      el.innerHTML =
        '<div class="ticket-head"><span class="ticket-id">' + escapeHtml(t.id) + '</span>' +
        '<span class="badge badge-gray">' + escapeHtml(INTENT_LABELS[t.intent] || t.intent) + '</span></div>' +
        '<div class="ticket-body">' + escapeHtml(t.user_message) + '</div>' +
        '<div class="ticket-time">' + escapeHtml((t.created_at || '').slice(0, 19).replace('T', ' ')) + '</div>';
      ticketsEl.appendChild(el);
    });
  } catch (e) {
    ticketsEl.innerHTML = '<div class="empty">工单加载失败</div>';
  }
}

async function checkHealth() {
  try {
    const data = await (await fetch('/health')).json();
    if (data.llm === 'mock') {
      backendBadge.textContent = '离线 Mock';
      backendBadge.className = 'badge badge-gray';
    } else {
      backendBadge.textContent = '真实模型';
      backendBadge.className = 'badge badge-green';
    }
  } catch (e) {
    backendBadge.textContent = '服务未连接';
    backendBadge.className = 'badge badge-red';
  }
}

// 事件绑定
sendBtn.addEventListener('click', sendMessage);
document.getElementById('new-session').addEventListener('click', newSession);
document.getElementById('toggle-sidebar').addEventListener('click', toggleSidebar);
document.getElementById('refresh-history').addEventListener('click', loadSessions);
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
});

// 初始化
setSessionId(sessionId);
checkHealth();
loadTickets();
loadSessions();
addMessage('assistant', '你好！我是优选商城智能客服，请问有什么可以帮您？', { intent: null });
