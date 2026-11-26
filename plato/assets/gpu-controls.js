"use strict";
const info = window.controlIndex;
const byId = (id) => document.getElementById(id);
const choice = (id, value, text) => byId(id).add(new Option(text, value));
const percent = (value) => value == null ? "—" : (100 * value).toFixed(1) + "%";
const interval = (metric) => metric.low == null ? "Unavailable" : percent(metric.low) + " to " + percent(metric.high);
const cache = new Map();
let worldCache, activeGroup;
function draw() {
  if (!worldCache) return;
  const group = activeGroup;
  const currentWorld = +byId("world").value;
  const lane = byId("lane").value;
  const variant = +byId("variant").value;
  const row = worldCache.find(r => r.world === currentWorld && r.variant === variant && r.lane === lane);
  if (!row) throw Error("Missing recorded selection");
  byId("prompt").textContent = row.prompt;
  byId("answer").textContent = row.answer || "⟨empty output⟩";
  byId("target").textContent = "Expected " + row.target + " · " + (row.correct ? "Correct" : "Incorrect");
  byId("source-detail").hidden = !row.source_prompt && !row.packet_sha256;
  byId("source").textContent = row.source_prompt || "";
  byId("packet").textContent = row.packet_sha256 ? "BF16 packet SHA-256 " + row.packet_sha256 : "";
  const metric = group.metrics[lane];
  byId("metric").textContent = "Across " + metric.worlds + " worlds, this condition scored " + metric.correct + " / " + metric.observations + " (" + percent(metric.mean) + "). " + metric.valid_answer + " answers were valid " + metric.answer_kind + " values.";
  byId("status").textContent = group.label + " · World " + currentWorld + " · " + info.categories[variant];
  for (const tr of byId("metrics").rows) tr.setAttribute("aria-current", String(tr.dataset.lane === lane));
}
function renderGroup() {
  const key = byId("dataset").value;
  activeGroup = info.groups[key];
  const group = activeGroup;
  const previousWorld = byId("world").value, previousLane = byId("lane").value;
  for (const id of ["world","lane","metrics"]) byId(id).replaceChildren();
  for (const world of group.world_ids) choice("world", world, world);
  for (const [id,metric] of Object.entries(group.metrics)) {
    choice("lane", id, metric.label);
    const tr = byId("metrics").insertRow();
    tr.dataset.lane = id;
    for (const text of [metric.label, metric.correct + " / " + metric.observations, percent(metric.mean), interval(metric)])
      tr.insertCell().textContent = text;
  }
  if (group.world_ids.includes(+previousWorld)) byId("world").value = previousWorld || group.world_ids[0];
  if (group.metrics[previousLane]) byId("lane").value = previousLane;
  byId("cohort").textContent = group.note;
  worldCache = null;
  if (cache.has(key)) { worldCache = cache.get(key); draw(); return; }
  byId("status").textContent = "Loading recorded answers…";
  const script = document.createElement("script");
  script.src = group.file;
  script.onerror = () => { byId("status").textContent = "Could not read the recordings. Download the raw data archive to inspect them."; };
  document.head.append(script);
}
window.receiveControlRows = (key, rows) => {
  cache.set(key,rows);
  if (byId("dataset").value === key) { worldCache = rows; draw(); }
};
byId("overview").textContent = info.statement;
for (const [key, group] of Object.entries(info.groups)) choice("dataset", key, group.label);
info.categories.forEach((name,i) => choice("variant",i,name));
byId("dataset").addEventListener("change", renderGroup);
for (const id of ["world","lane","variant"]) byId(id).addEventListener("change", draw);
renderGroup();
