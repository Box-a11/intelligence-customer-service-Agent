const INTENT_LABELS = {
  qa: '问答',
  consultation: '咨询',
  complaint: '投诉',
  unclear: '模糊',
};

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// 数值自动压缩：1,284 / 12.9K
function fmtNum(n) {
  if (n >= 10000) return (n / 1000).toFixed(1) + 'K';
  return n.toLocaleString('en-US');
}

// 延迟：8200 -> 8.2s；820 -> 820ms
function fmtLatency(ms) {
  if (ms >= 1000) return (ms / 1000).toFixed(1) + 's';
  return Math.round(ms) + 'ms';
}

// 渲染水平条形图：label + 轨道 + 填充 + 数值
function renderBars(el, items) {
  if (!items.length) {
    el.innerHTML = '<div class="empty">暂无数据</div>';
    return;
  }
  const max = Math.max(...items.map((i) => i.count));
  el.innerHTML = items.map((i) => {
    const pct = max > 0 ? Math.round((i.count / max) * 100) : 0;
    return (
      '<div class="bar-row">' +
        '<span class="bar-label" title="' + escapeHtml(i.label) + '">' + escapeHtml(i.label) + '</span>' +
        '<div class="bar-track"><div class="bar-fill" style="width:' + pct + '%"></div></div>' +
        '<span class="bar-value">' + fmtNum(i.count) + '</span>' +
      '</div>'
    );
  }).join('');
}

// 渲染键值对列表
function renderKV(el, pairs) {
  el.innerHTML = pairs.map(([k, v]) =>
    '<div class="kv-row"><span class="kv-key">' + escapeHtml(k) + '</span>' +
    '<span class="kv-val">' + escapeHtml(v) + '</span></div>'
  ).join('');
}

function renderRecent(rows) {
  const tbody = document.getElementById('recent-body');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty">暂无请求</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map((r) => {
    const srcs = (r.sources || []);
    const srcText = srcs.length ? escapeHtml(srcs.slice(0, 3).join('、') + (srcs.length > 3 ? '…' : '')) : '—';
    const result = r.has_ticket
      ? '<span class="badge badge-ticket">转人工</span>'
      : (r.needs_clarification ? '<span class="badge badge-gray">澄清</span>' : '<span class="badge badge-green">完成</span>');
    return (
      '<tr>' +
        '<td class="t-time">' + escapeHtml((r.time || '').slice(11, 19)) + '</td>' +
        '<td class="t-msg" title="' + escapeHtml(r.message || '') + '">' + escapeHtml(r.message || '') + '</td>' +
        '<td><span class="badge badge-intent">' + escapeHtml(INTENT_LABELS[r.intent] || r.intent || '—') + '</span></td>' +
        '<td class="t-num">' + fmtLatency(r.latency_ms) + '</td>' +
        '<td class="t-src" title="' + srcText + '">' + srcText + '</td>' +
        '<td>' + result + '</td>' +
      '</tr>'
    );
  }).join('');
}

async function loadMetrics() {
  try {
    const d = await (await fetch('/metrics')).json();

    document.getElementById('kpi-requests').textContent = fmtNum(d.counts.requests);
    document.getElementById('kpi-sessions').textContent = fmtNum(d.counts.sessions);
    document.getElementById('kpi-tickets').textContent = fmtNum(d.counts.tickets);
    document.getElementById('kpi-latency').textContent = fmtLatency(d.latency.avg_ms);

    // 意图分布（固定顺序 + 数量降序）
    const intents = d.intents || {};
    const intentItems = ['qa', 'consultation', 'complaint', 'unclear']
      .map((k) => ({ label: INTENT_LABELS[k], count: intents[k] || 0 }))
      .sort((a, b) => b.count - a.count);
    renderBars(document.getElementById('intent-bars'), intentItems);

    // 检索来源 Top
    const sourceItems = (d.retrieval.top_sources || []).map((s) => ({ label: s.source, count: s.count }));
    renderBars(document.getElementById('source-bars'), sourceItems);

    // 系统信息
    renderKV(document.getElementById('sys-info'), [
      ['LLM 后端', d.system.llm === 'mock' ? '离线 Mock' : '真实模型'],
      ['模型', d.system.llm_model],
      ['向量后端', d.system.embed_backend],
      ['分片路由', d.system.routing ? '已启用' : '关闭'],
      ['版本', d.system.version],
    ]);

    // 工单与延迟
    const tb = d.tickets_breakdown;
    renderKV(document.getElementById('ticket-info'), [
      ['投诉工单', tb.complaint],
      ['转人工工单', tb.escalate],
      ['待处理 / 已关闭', tb.open + ' / ' + tb.closed],
      ['平均延迟', fmtLatency(d.latency.avg_ms)],
      ['最大延迟', fmtLatency(d.latency.max_ms)],
      ['最小延迟', fmtLatency(d.latency.min_ms)],
    ]);

    renderRecent(d.recent_requests || []);
  } catch (e) {
    document.getElementById('kpi-requests').textContent = '—';
    document.getElementById('recent-body').innerHTML =
      '<tr><td colspan="6" class="empty">大盘数据加载失败：' + escapeHtml(e.message) + '</td></tr>';
  }
}

// 初始化 + 每 5 秒自动刷新
loadMetrics();
setInterval(loadMetrics, 5000);
