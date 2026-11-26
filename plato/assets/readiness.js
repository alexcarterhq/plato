"use strict";
const {summary: info, rows, labels} = window.readinessData;
const el = id => document.getElementById(id);
const set = (id, text) => { el(id).textContent = text; };
const fmt = n => n === null ? "unavailable" : (100 * n).toFixed(1) + "%";
const query = new URL(location.href).searchParams;
function options(id, entries, value) {
  const select = el(id);
  select.replaceChildren(...entries.map(([v,t]) => new Option(t,String(v))));
  if (entries.some(([v]) => String(v) === String(value))) select.value = String(value);
}
function line(parent, tag, cls, text) {
  const span=document.createElement(tag); span.className=cls; span.textContent=text; parent.appendChild(span);
}
set("decision",info.statement);
set("scope",info.worlds + " of 64 validation worlds · CPU FP32 · Cross-model held-out test sealed");
for (const [key,title] of Object.entries(labels)) {
  const stats=info.lanes[key], div=document.createElement("div"); div.className="score";
  line(div,"strong","number",fmt(stats.mean)); line(div,"span","lane",title);
  let number=stats.correct + " / " + stats.observations + " correct";
  number += stats.low === null ? " · One world" : " · 95% CI " + fmt(stats.low) + "–" + fmt(stats.high);
  line(div,"span","denominator",number); el("metrics").appendChild(div);
}
const table=new Map(rows.map(r => [[r.world,r.variant,r.condition,r.rulebook].join("/"),r]));
options("world",info.world_ids.map(w=>[w,"World "+w]),query.get("world"));
const variants=rows.filter(r=>r.world===info.world_ids[0] && r.condition==="sender").sort((a,b)=>a.variant-b.variant);
options("variant",variants.map(r=>[r.variant,r.category]),query.get("variant"));
options("book",[[0,"Rulebook 1"],[1,"Rulebook 2"]],query.get("book"));
function render() {
  const world=Number(el("world").value), v=Number(el("variant").value), book=Number(el("book").value);
  for (const [arm,b] of [["sender",null],["full",book]]) {
    const r=table.get([world,v,arm,b].join("/"));
    set(arm+"-target",r.target);
    set(arm+"-outcome", r.correct ? "Correct · raw completion" : "Incorrect · raw completion");
    el(arm+"-outcome").className=r.correct?"correct":"incorrect";
    set(arm+"-answer",r.answer || "⟨empty completion⟩"); set(arm+"-prompt",r.prompt);
  }
  set("status","World "+world+" · "+variants[v].category+" · Rulebook "+(book+1));
  const u=new URL(location.href); u.searchParams.set("world",String(world));u.searchParams.set("variant",String(v));u.searchParams.set("book",String(book));
  try { history.replaceState(null,"",u); } catch (_) {}
}
["world","variant","book"].forEach(id=>el(id).addEventListener("change",render)); render();
