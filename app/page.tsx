"use client";

import { useRef, useState } from "react";

const findings = [
  { level: "Critical", title: "Exact duplicates across splits", detail: "23 images appear in both training and test sets", count: 23, icon: "≋" },
  { level: "High", title: "Near-duplicate clusters", detail: "Crops, resizes, or compressed copies across splits", count: 47, icon: "◫" },
  { level: "High", title: "Conflicting labels", detail: "Similar images assigned to different classes", count: 8, icon: "⇄" },
  { level: "Medium", title: "Class distribution drift", detail: "Test split is under-represented for 2 classes", count: 2, icon: "⌁" },
];

export default function Home() {
  const [active, setActive] = useState("Overview");
  const [selected, setSelected] = useState(0);
  const [toast, setToast] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  function notify(message: string) { setToast(message); window.setTimeout(() => setToast(""), 2600); }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">S</span><span>SplitGuard</span></div>
        <nav aria-label="Primary navigation">
          <p className="nav-label">WORKSPACE</p>
          {["Overview", "Findings", "Similarity map", "Impact experiment"].map((item, i) => (
            <button key={item} className={active === item ? "nav-item active" : "nav-item"} onClick={() => { setActive(item); notify(`${item} view selected`); }}>
              <span>{["▦", "⚑", "⌘", "↗"][i]}</span>{item}{item === "Findings" && <b>80</b>}
            </button>
          ))}
          <p className="nav-label datasets">DATASETS</p>
          <button className="dataset active-dataset"><span className="dataset-dot" />animals-10 <small>Scanning complete</small></button>
          <button className="add-dataset" onClick={() => inputRef.current?.click()}>＋ Add dataset</button>
          <input ref={inputRef} type="file" multiple hidden onChange={(e) => e.target.files?.length && notify(`${e.target.files.length} files ready to import`)} />
        </nav>
        <div className="sidebar-bottom"><button className="nav-item"><span>?</span>Documentation</button><div className="profile"><div className="avatar">CO</div><div><strong>Clarence</strong><small>Local workspace</small></div><button>•••</button></div></div>
      </aside>

      <section className="content">
        <header><div><div className="eyebrow"><span className="status-dot" /> AUDIT COMPLETE <span>•</span> AUG 22, 2026</div><h1>Dataset integrity overview</h1><p>animals-10 <span>•</span> 26,179 images <span>•</span> 10 classes</p></div><div className="header-actions"><button className="secondary" onClick={() => notify("Report export prepared")}>⇩ Export report</button><button className="primary" onClick={() => { setActive("Impact experiment"); notify("Impact experiment ready to configure"); }}>Run impact experiment <span>→</span></button></div></header>

        <section className="hero-grid">
          <article className="score-card"><div className="card-title"><span>Integrity score</span><button aria-label="About integrity score">?</button></div><div className="score-wrap"><div className="score-ring"><div><strong>62</strong><small>/ 100</small></div></div><div className="score-copy"><span className="risk-pill">HIGH RISK</span><h2>Your evaluation may be unreliable.</h2><p>Cross-split leakage and label conflicts could be inflating reported performance.</p><button onClick={() => { setActive("Findings"); document.getElementById("findings")?.scrollIntoView({ behavior: "smooth" }); }}>Review 80 findings →</button></div></div><div className="score-legend"><span><i className="bad" />Leakage risk <b>High</b></span><span><i className="warn" />Label integrity <b>Fair</b></span><span><i className="good" />Distribution <b>Good</b></span></div></article>
          <article className="impact-card"><div className="impact-head"><div><span className="mini-label">IMPACT EXPERIMENT</span><h2>Your 94% may actually be 82%.</h2></div><span className="verified">● VERIFIED RUN</span></div><div className="metric-row"><div><span>ORIGINAL EVALUATION</span><strong>94.2<small>%</small></strong></div><span className="arrow">→</span><div><span>AFTER LEAKAGE REMOVAL</span><strong className="clean-score">81.7<small>%</small></strong></div></div><div className="difference"><span>Observed difference</span><strong>−12.5 percentage points</strong></div><div className="confidence"><span><i />3 controlled runs</span><span><i />Same model &amp; seed policy</span><button onClick={() => { setActive("Impact experiment"); notify("Opening experiment details"); }}>View experiment →</button></div></article>
        </section>

        <section className="lower-grid" id="findings">
          <article className="findings-card"><div className="section-head"><div><h2>What needs attention</h2><p>Ranked by potential impact on your evaluation</p></div><button onClick={() => setActive("Findings")}>View all findings →</button></div><div className="finding-list">{findings.map((item, index) => <button key={item.title} className={selected === index ? "finding selected" : "finding"} onClick={() => setSelected(index)}><span className={`finding-icon ${item.level.toLowerCase()}`}>{item.icon}</span><span className="finding-copy"><span className={`level ${item.level.toLowerCase()}`}>{item.level}</span><strong>{item.title}</strong><small>{item.detail}</small></span><b>{item.count}</b><span className="chevron">›</span></button>)}</div></article>
          <article className="split-card"><div className="section-head"><div><h2>Split health</h2><p>Cross-split contamination</p></div><button aria-label="More options">•••</button></div><div className="split-chart"><div className="venn train"><span>TRAIN</span><b>18,325</b></div><div className="venn test"><span>TEST</span><b>5,236</b></div><div className="overlap"><strong>70</strong><span>LEAKED</span></div></div><div className="split-stats"><div><span>TRAIN ↔ TEST</span><strong>53</strong><small>items overlap</small></div><div><span>TRAIN ↔ VALIDATION</span><strong>17</strong><small>items overlap</small></div></div><button className="map-link" onClick={() => { setActive("Similarity map"); notify("Similarity map selected"); }}>Explore similarity map <span>→</span></button></article>
        </section>
        <footer><span><i /> Analysis completed in 2m 14s</span><span>SHA-256 + pHash + vision embeddings</span></footer>
      </section>{toast && <div className="toast" role="status">✓ {toast}</div>}
    </main>
  );
}
