import assert from "node:assert/strict";
import { readFile, access } from "node:fs/promises";
import test from "node:test";

const root=new URL("../",import.meta.url);
const read=(path)=>readFile(new URL(path,root),"utf8");

test("build emits the React Vite application",async()=>{
  const html=await read("dist/index.html");
  assert.match(html,/SplitGuard/);assert.match(html,/id="root"/);assert.match(html,/favicon\.svg/);await access(new URL("dist/assets",root));
});

test("frontend has no dataset-specific accuracy fallback or automatic experiment",async()=>{
  const source=await read("src/App.tsx");
  assert.doesNotMatch(source,/rocks-impact-results|archive-impact-results|experimentKey/);
  assert.doesNotMatch(source,/runExperiment\(completed,files\)/);
  assert.match(source,/VITE_SPLITGUARD_API_URL/);assert.match(source,/I agree/);assert.match(source,/three-seed/);
});

test("audit and cleaning preserve split-aware leakage controls",async()=>{
  const audit=await read("backend/audit_dataset.py");const prep=await read("backend/prepare_impact_datasets.py");const train=await read("backend/run_impact_experiment.py");
  assert.match(audit,/"split":split/);assert.match(audit,/BKNode/);assert.match(audit,/cross_split/);
  assert.match(prep,/same_test_set/);assert.match(prep,/matches held-out evaluation image/);assert.match(prep,/baseline_original/);
  assert.match(train,/accuracy_ci95/);assert.match(train,/default=\[42,1337,2027\]/);assert.match(train,/FixedFolder/);
});

test("worker is configurable, queued, cancellable and disposable",async()=>{
  const api=await read("backend/api.py");
  assert.match(api,/SPLITGUARD_CORS_ORIGINS/);assert.match(api,/http:\/\/localhost:5173/);assert.match(api,/queue\.Queue/);assert.match(api,/RETENTION/);assert.match(api,/\/cancel/);assert.match(api,/@app\.delete/);
});
