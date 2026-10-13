const EXAMPLES = [
  "一个男人潜入别人的梦里偷取想法，最后陀螺一直在转",
  "核战之后的废土上，有人开着改装卡车抢水",
  "穷画家和富家女在撞上冰山的巨轮上相爱",
  "小女孩的父母变成了猪，她走进神灵世界想把他们变回来",
  "复制人会梦见电子羊，雨一直下在未来城市",
  "奥本海默组织科学家制造原子弹",
];

const queryInput = document.querySelector("#query");
const form = document.querySelector("#search-form");
const button = document.querySelector("#search-button");
const stage = document.querySelector("#stage");
const statusBox = document.querySelector("#status");
const results = document.querySelector("#results");
const denseList = document.querySelector("#dense-list");
const bm25List = document.querySelector("#bm25-list");
const chips = document.querySelector("#chips");
const hint = document.querySelector("#hint");
const canvas = document.querySelector("#map");
const mapTip = document.querySelector("#map-tip");
const universeNote = document.querySelector("#universe-note");

const student = window.STUDENT || { id: "请填写学号", name: "请填写姓名" };
document.querySelector("#student-badge").textContent = `${student.id}　${student.name}`;

let placeholderIndex = 0;
let universe = null;
let queryPoint = null;
let highlighted = new Set();

function setPlaceholder() {
  if (document.activeElement === queryInput || queryInput.value) return;
  queryInput.placeholder = EXAMPLES[placeholderIndex % EXAMPLES.length];
  placeholderIndex += 1;
}
setPlaceholder();
setInterval(setPlaceholder, 3200);

EXAMPLES.forEach((text) => {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.textContent = text.length > 18 ? `${text.slice(0, 18)}…` : text;
  chip.title = text;
  chip.addEventListener("click", () => {
    queryInput.value = text;
    form.requestSubmit();
  });
  chips.appendChild(chip);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (query.length < 2) {
    showStatus("再多写几个字。太短的时候，语义和关键词都很难判断你在找哪部电影。");
    results.hidden = true;
    return;
  }
  button.disabled = true;
  button.textContent = "寻找中";
  showStatus("正在比较句向量和 BM25…");
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 6 }),
    });
    const payload = await response.json();
    if (!response.ok) {
      showStatus(payload.detail || "检索失败");
      results.hidden = true;
      return;
    }
    renderResults(payload);
  } catch (error) {
    showStatus("检索服务没有响应。请确认已经启动应用。");
  } finally {
    button.disabled = false;
    button.textContent = "找回";
  }
});

function showStatus(text) {
  statusBox.hidden = false;
  statusBox.textContent = text;
}

function renderResults(payload) {
  stage.classList.add("has-results");
  results.hidden = false;
    const top = payload.dense[0];
    if (!top) {
      showStatus("没有返回结果。");
    } else if (!payload.confident) {
      showStatus(`最接近的是「${top.title}」（${top.year}），优势不大。描述可能太笼统，或这部电影不在 1901–2017 的片库里。右侧 BM25 只看字面：中文问句通常没有结果，英文关键词则可能对上。`);
    } else {
      showStatus(`语义检索更相信「${top.title}」（${top.year}）。右侧 BM25 用原始问句做字面匹配，用来对照。片库大约收到 2017 年，更晚的电影不会出现。`);
    }
  highlighted = new Set(payload.dense.slice(0, 5).map((item) => item.id));
  queryPoint = payload.query_point;
  denseList.innerHTML = payload.dense.map((item) => cardHtml(item, "dense")).join("");
  bm25List.innerHTML = payload.bm25.length
    ? payload.bm25.map((item) => cardHtml(item, "bm25")).join("")
    : `<div class="empty">BM25 没有找到共享关键词。中文情节对英文剧情几乎对不上字面，这正是关键词检索的盲区。</div>`;
  bindPosters(denseList);
  bindPosters(bm25List);
  drawMap();
}

function cardHtml(item, kind) {
  const scoreLabel = kind === "dense"
    ? `相关度 ${item.score.toFixed(2)}`
    : `BM25 ${item.score.toFixed(2)}`;
  const initial = (item.title || "?").slice(0, 1);
  return `
    <article class="card ${kind}">
      <img class="poster fallback" alt="" data-wiki="${escapeAttr(item.wiki)}" data-initial="${escapeAttr(initial)}" />
      <div>
        <h3>${escapeHtml(item.title)} <span class="meta">${escapeHtml(item.year)}</span></h3>
        <p class="meta">${escapeHtml(item.origin)} · ${escapeHtml(item.genre || "类型未知")} · ${escapeHtml(item.director || "导演未知")}</p>
        <p class="plot">${escapeHtml(item.evidence || item.plot || "")}</p>
        <span class="score">${scoreLabel}${item.phrases && item.phrases.length ? " · 短语命中" : ""}</span>
      </div>
    </article>`;
}

function bindPosters(root) {
  root.querySelectorAll(".poster").forEach((img) => {
    const initial = img.dataset.initial || "?";
    paintFallback(img, initial);
    const wiki = img.dataset.wiki;
    if (!wiki) return;
    fetch(`/api/poster?wiki=${encodeURIComponent(wiki)}`)
      .then((response) => response.json())
      .then((payload) => {
        if (!payload.image) return;
        const photo = new Image();
        photo.onload = () => {
          img.src = payload.image;
          img.classList.remove("fallback");
        };
        photo.src = payload.image;
      })
      .catch(() => {});
  });
}

function paintFallback(img, initial) {
  const canvas = document.createElement("canvas");
  canvas.width = 184;
  canvas.height = 264;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createLinearGradient(0, 0, 180, 260);
  gradient.addColorStop(0, "#9a3412");
  gradient.addColorStop(1, "#1c1915");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "white";
  ctx.font = "64px Fraunces, serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(initial, canvas.width / 2, canvas.height / 2);
  img.src = canvas.toDataURL();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}
function escapeAttr(value) {
  return escapeHtml(value).replaceAll('"', "&quot;");
}

async function loadUniverse() {
  const response = await fetch("/api/universe");
  universe = await response.json();
  const ratio = (universe.explained || []).map((v) => `${(v * 100).toFixed(1)}%`).join(" + ");
  universeNote.textContent = `PCA 解释方差 ${ratio} · 图中是按年代抽样的 ${universe.points.length} 部`;
  drawMap();
}

function drawMap() {
  if (!universe) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  const points = universe.points;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  if (queryPoint) {
    xs.push(queryPoint.x);
    ys.push(queryPoint.y);
  }
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const pad = 28;
  const sx = (width - pad * 2) / (maxX - minX || 1);
  const sy = (height - pad * 2) / (maxY - minY || 1);
  const project = (x, y) => [pad + (x - minX) * sx, height - pad - (y - minY) * sy];

  points.forEach((point) => {
    const [x, y] = project(point.x, point.y);
    ctx.beginPath();
    ctx.fillStyle = highlighted.has(point.id) ? "#9a3412" : "rgba(80,70,60,0.28)";
    ctx.arc(x, y, highlighted.has(point.id) ? 5 : 2.2, 0, Math.PI * 2);
    ctx.fill();
  });
  if (queryPoint) {
    const [x, y] = project(queryPoint.x, queryPoint.y);
    ctx.beginPath();
    ctx.strokeStyle = "#9a3412";
    ctx.lineWidth = 2;
    ctx.arc(x, y, 9, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = "#9a3412";
    ctx.font = "14px DM Sans";
    ctx.fillText("这次查询", x + 12, y - 8);
  }
  canvas._project = project;
}

canvas.addEventListener("mousemove", (event) => {
  if (!universe || !canvas._project) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const mx = (event.clientX - rect.left) * scaleX;
  const my = (event.clientY - rect.top) * scaleY;
  let nearest = null;
  let best = 14;
  for (const point of universe.points) {
    const [x, y] = canvas._project(point.x, point.y);
    const distance = Math.hypot(x - mx, y - my);
    if (distance < best) {
      best = distance;
      nearest = point;
    }
  }
  mapTip.textContent = nearest
    ? `${nearest.title}（${nearest.year}，${nearest.origin}）`
    : "悬停查看片名。搜索之后，查询会变成图上的星。";
});

loadUniverse().catch(() => {
  universeNote.textContent = "电影宇宙暂时没有载入";
});
