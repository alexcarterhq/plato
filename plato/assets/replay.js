"use strict";
const manifest = window.platoManifest;
const byId = id => document.getElementById(id);
const controls = Object.fromEntries(["sender", "world", "variant", "book", "method"].map(id => [id, byId(id)]));
const cache = new Map();
let active = null;
let requested = "";
const text = (id, value) => { byId(id).textContent = value; };
const options = (select, entries, preferred) => {
  select.replaceChildren(...entries.map(([value, label]) => {
    const option = new Option(label, String(value));
    return option;
  }));
  if (entries.some(([value]) => String(value) === String(preferred))) select.value = String(preferred);
};
const label = value => value.charAt(0).toUpperCase() + value.slice(1);
const pp = value => value === null ? "—" : (100 * value).toFixed(1);
const interval = s => s.mean === null ? "unavailable" : s.low === null
  ? pp(s.mean) + " pp" : pp(s.mean) + " pp [" + pp(s.low) + ", " + pp(s.high) + "]";
function showSummary(sender) {
  const group = manifest.result.groups.test[sender];
  const c = group.coverage;
  text("aggregate-intro", c.complete_worlds + " complete worlds of " + c.worlds
    + " recorded. Differences compare matched and mismatched packets on the same cases.");
  const cards = ["factual", "contextual"].map(bridge => {
    const div = document.createElement("div");
    div.className = "readout";
    const name = document.createElement("strong");
    name.textContent = manifest.conditions[bridge];
    div.appendChild(name);
    for (const book of [0, 1]) {
      const detail = document.createElement("span");
      detail.textContent = "Rulebook " + (book + 1) + " · " + interval(group.books[book][bridge + "_lift"]);
      div.appendChild(detail);
    }
    return div;
  });
  byId("readouts").replaceChildren(...cards);
}
function fill(rows, keepVariant) {
  const book = Number(controls.book.value);
  const labels = new Map();
  for (const row of rows.filter(r => r.rulebook === book)) labels.set(row.variant, row.category || String(row.variant));
  options(controls.variant, [...labels.entries()].sort((a, b) => a[0] - b[0]), keepVariant);
  render();
}
function render() {
  if (!active) return;
  const book = Number(controls.book.value), variant = Number(controls.variant.value);
  const rows = active.rows.filter(r => r.variant === variant && r.rulebook === book);
  const selected = rows.find(r => r.condition === controls.method.value);
  const wrongPacket = selected?.condition.endsWith("_wrong");
  text("sender-context", wrongPacket ? "Delivered packet for alternate category " + selected.wrong_category
    : "Private input to the sending model");
  text("sender-prompt", selected ? active.prompts[selected.sender_prompt] || "Prompt unavailable." : "Condition unavailable.");
  text("receiver-prompt", selected ? active.prompts[selected.receiver_prompt] || "Prompt unavailable." : "Condition unavailable.");
  text("target", selected?.target ?? "unavailable");
  const cells = [];
  for (const [key, title] of Object.entries(manifest.conditions)) {
    const record = rows.find(r => r.condition === key);
    const row = document.createElement("tr");
    if (key === controls.method.value) row.setAttribute("aria-current", "true");
    const wrong = key.endsWith("_wrong") && record?.wrong_category ? " (" + record.wrong_category + ")" : "";
    const fields = [title + wrong, record ? (record.answer || "⟨empty⟩") : "—",
      record ? (record.correct ? "Correct" : "Incorrect") : "Missing",
      Number.isFinite(record?.packet_bytes) ? record.packet_bytes.toLocaleString() + " B" : "—"];
    for (let i = 0; i < fields.length; i++) {
      const cell = document.createElement("td");
      cell.textContent = fields[i];
      if (i === 1) cell.className = "answer";
      if (i === 2) cell.className = "outcome " + (record?.correct ? "correct" : "incorrect");
      row.appendChild(cell);
    }
    cells.push(row);
  }
  byId("rows").replaceChildren(...cells);
  const url = new URL(location.href);
  url.searchParams.set("sender", active.sender); url.searchParams.set("world", String(active.world));
  url.searchParams.set("book", String(book)); url.searchParams.set("variant", String(variant));
  url.searchParams.set("method", controls.method.value);
  try { history.replaceState(null, "", url); } catch (_) { /* Restricted local-file browser */ }
}
const initial = new URL(location.href).searchParams;
let initialLoad = true;
function showWorld(world) {
  active = world;
  const books = [...new Set(world.rows.map(r => r.rulebook))].sort();
  const book = initialLoad ? initial.get("book") : controls.book.value;
  const variant = initialLoad ? initial.get("variant") : controls.variant.value;
  initialLoad = false;
  options(controls.book, books.map(i => [i, "Rulebook " + (i + 1)]), book);
  fill(world.rows, variant);
  text("status", "");
  byId("app").removeAttribute("aria-busy");
}
window.platoWorld = world => {
  cache.set(world.key, world);
  if (world.key === requested) showWorld(world);
};
function loadWorld() {
  const key = controls.sender.value + "-" + controls.world.value;
  requested = key;
  if (cache.has(key)) { showWorld(cache.get(key)); return; }
  active = null;
  byId("app").setAttribute("aria-busy", "true");
  text("sender-prompt", "Loading…"); text("receiver-prompt", "Loading…"); text("target", "…");
  byId("rows").replaceChildren();
  text("status", "Loading recorded world…");
  const script = document.createElement("script");
  script.src = "worlds/" + key + ".js";
  script.onload = () => script.remove();
  script.onerror = () => {
    script.remove();
    if (key !== requested) return;
    byId("app").removeAttribute("aria-busy");
    text("status", "This recording could not be loaded. Check the worlds directory or select another world.");
  };
  document.head.appendChild(script);
}
function changeSender(preferredWorld) {
  const sender = controls.sender.value;
  const worlds = manifest.worlds.filter(x => x.sender === sender);
  options(controls.world, worlds.map(w => [w.world, "World " + w.world]), preferredWorld);
  showSummary(sender); loadWorld();
}
if (manifest?.result?.diagnostics && !manifest.worlds.length) {
  for (const [sender, diag] of Object.entries(manifest.result.diagnostics)) {
    const p = document.createElement("p");
    const parts = [label(sender) + " sender."];
    if (diag.factual) parts.push("Factual screening gate " + (diag.factual.gate_passed ? "passed." : "was not met."));
    if (diag.validation) parts.push("Validation controls " + (diag.validation.controls_valid ? "passed." : "were not met."));
    if (diag.diagnostic?.reason) parts.push(String(diag.diagnostic.reason));
    p.textContent = parts.join(" ");
    byId("diagnostic-detail").appendChild(p);
    byId("diagnostics").hidden = false;
  }
}
if (!manifest?.worlds?.length && manifest?.result?.groups?.validation) {
  const figure = byId("validation-figure");
  figure.querySelector("img").src = "validation-accuracy.svg";
  figure.hidden = false;
  byId("diagnostics").hidden = false;
}
if (!manifest?.worlds?.length) {
  byId("empty").hidden = false;
  if (!manifest) {
    byId("empty").querySelector("h2").textContent = "Recording manifest unavailable";
    byId("empty").querySelector("p").textContent = "Open the full release directory to load the report data.";
  }
} else {
  byId("app").hidden = false;
  byId("cond-figure").src = "condition-accuracy.svg";
  byId("lift-figure").src = "paired-lift.svg";
  const senders = ["qwen", "pythia"].filter(id => manifest.worlds.some(w => w.sender === id));
  const models = {qwen:"Qwen2.5 · primary", pythia:"Pythia · replication"};
  options(controls.sender, senders.map(value => [value, models[value]]), initial.get("sender"));
  options(controls.method, Object.entries(manifest.conditions).filter(([key]) => key !== "sender"), initial.get("method") ?? "factual");
  controls.method.addEventListener("change", render);
  controls.sender.addEventListener("change", () => changeSender(controls.world.value));
  controls.world.addEventListener("change", loadWorld);
  controls.book.addEventListener("change", () => { if (active) fill(active.rows, controls.variant.value); });
  controls.variant.addEventListener("change", render);
  changeSender(initial.get("world"));
}
