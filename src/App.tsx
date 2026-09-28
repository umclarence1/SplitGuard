import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";

type Item = { name:string; path:string; split:string; label:string; size:number; hash:string; visual:string; width:number; height:number };
type Finding = { kind:"Exact duplicate"|"Near duplicate"|"Label conflict"; severity:"Critical"|"High"; a:Item; b:Item; similarity:number };
type Report = { name:string; items:Item[]; findings:Finding[]; splits:Record<string,number>; classes:Record<string,number>; exact:number; near:number; conflicts:number; score:number; duration:number };
type Impact = { baseline:{test:{accuracy:number;accuracy_std:number;accuracy_ci95:number;macro_f1:number}}; cleaned:{test:{accuracy:number;accuracy_std:number;accuracy_ci95:number;macro_f1:number}}; observed_difference_percentage_points:number };
type ExperimentJob = { id:string; status:string; message:string; progress:number; result?:Impact; preparation?:{removed_training_images:number;source_split:string}; owner_token?:string };
const IMAGE_TYPES = new Set(["image/jpeg","image/png","image/webp","image/gif","image/bmp"]);
const API_URL=(import.meta.env.VITE_SPLITGUARD_API_URL||"http://localhost:8000").replace(/\/$/,"");
const API_KEY=import.meta.env.VITE_SPLITGUARD_API_KEY||"";
const OWNER_HEADER="X-SplitGuard-Owner-Token";
const SESSION_KEY="splitguard.session.v1";
type SavedSession={report:Report|null;job:{id:string;owner_token?:string}|null};

function loadSession():SavedSession{
  try{const raw=sessionStorage.getItem(SESSION_KEY);if(raw)return JSON.parse(raw) as SavedSession}catch{/* storage unavailable or corrupt */}
  return {report:null,job:null};
}
function saveSession(report:Report|null,job:ExperimentJob|null){
  const jobRef=job?.id?{id:job.id,owner_token:job.owner_token}:null;
  const slim=report?({...report,items:report.items.map(i=>({size:i.size}))} as unknown as Report):null;
  for(const candidate of [report,slim]){try{sessionStorage.setItem(SESSION_KEY,JSON.stringify({report:candidate,job:jobRef}));return}catch{/* over quota: retry with per-file details dropped */}}
  try{sessionStorage.removeItem(SESSION_KEY)}catch{/* ignore */}
}

async function mapWithConcurrency<T,R>(items:T[],limit:number,fn:(item:T,index:number)=>Promise<R>):Promise<R[]>{
  const results:R[]=new Array(items.length);let next=0;
  async function worker(){while(next<items.length){const index=next++;results[index]=await fn(items[index],index)}}
  await Promise.all(Array.from({length:Math.min(limit,items.length)},worker));
  return results;
}

function parsePath(file: File) {
  const path = (file as File & { webkitRelativePath?:string }).webkitRelativePath || file.name;
  const parts = path.split("/");
  const known = new Set(["train","training","val","valid","validation","test","testing"]);
  const splitIndex = parts.findIndex(p => known.has(p.toLowerCase()));
  const raw = splitIndex >= 0 ? parts[splitIndex].toLowerCase() : "unsplit";
  const split = raw === "training" ? "train" : (["val","valid"].includes(raw) ? "validation" : raw === "testing" ? "test" : raw);
  const label = splitIndex >= 0 && parts[splitIndex+1] ? parts[splitIndex+1] : parts.length > 1 ? parts[parts.length-2] : "unlabeled";
  return { path, split, label, root:parts[0] || "dataset" };
}
async function sha256(file:File) { const bytes = await file.arrayBuffer(); const digest = await crypto.subtle.digest("SHA-256",bytes); return [...new Uint8Array(digest)].map(b=>b.toString(16).padStart(2,"0")).join(""); }
async function visualHash(file:File) {
  const bitmap = await createImageBitmap(file); const canvas = document.createElement("canvas"); canvas.width=9; canvas.height=8;
  const ctx=canvas.getContext("2d",{willReadFrequently:true})!; ctx.drawImage(bitmap,0,0,9,8); const data=ctx.getImageData(0,0,9,8).data; const light:number[]=[];
  for(let i=0;i<data.length;i+=4)light.push(Math.round(data[i]*.299+data[i+1]*.587+data[i+2]*.114));
  const bits:string[]=[];for(let y=0;y<8;y++)for(let x=0;x<8;x++)bits.push(light[y*9+x+1]>light[y*9+x]?"1":"0");
  return { hash:bits.join(""), width:bitmap.width, height:bitmap.height };
}
function distance(a:string,b:string){let d=0;for(let i=0;i<a.length;i++)if(a[i]!==b[i])d++;return d}
function nearCandidates(items:Item[]){
  const bands=new Map<string,number[]>();const pairs:[number,number][]=[];const seen=new Set<string>();const cuts=[[0,13],[13,26],[26,39],[39,52],[52,64]];
  items.forEach((item,index)=>{const candidates=new Set<number>();cuts.forEach(([a,b],band)=>{const key=`${band}:${item.visual.slice(a,b)}`;(bands.get(key)||[]).forEach(x=>candidates.add(x))});candidates.forEach(other=>{const key=`${other}:${index}`;if(!seen.has(key)){seen.add(key);pairs.push([other,index])}});cuts.forEach(([a,b],band)=>{const key=`${band}:${item.visual.slice(a,b)}`;bands.set(key,[...(bands.get(key)||[]),index])})});return pairs;
}
function estimateExperiment(report:Report){
  const totalGb=report.items.reduce((sum,item)=>sum+item.size,0)/(1024**3);
  const classCount=Object.keys(report.classes).length;
  const minutes=Math.max(4,report.items.length/160+totalGb*2+classCount*.04);
  const low=Math.max(3,Math.round(minutes*.75));
  const high=Math.max(low+2,Math.round(minutes*1.6));
  return `${low} to ${high} minutes`;
}

export default function Home(){
  const [saved]=useState(loadSession);
  const input=useRef<HTMLInputElement>(null); const [report,setReport]=useState<Report|null>(saved.report); const [selectedFiles,setSelectedFiles]=useState<File[]>([]); const [job,setJob]=useState<ExperimentJob|null>(saved.job?{id:saved.job.id,owner_token:saved.job.owner_token,status:"resuming",message:"Reconnecting to your experiment…",progress:0}:null); const [phase,setPhase]=useState<"idle"|"scan"|"done">(saved.report?"done":"idle"); const [progress,setProgress]=useState(0); const [error,setError]=useState(""); const [filter,setFilter]=useState("All");
  const visible=useMemo(()=>report?.findings.filter(f=>filter==="All"||f.kind===filter)||[],[report,filter]);
  useEffect(()=>{if(report)requestAnimationFrame(()=>window.scrollTo({top:0,left:0,behavior:"instant"}))},[report]);
  // Only id/owner_token are persisted, so status/progress changes shouldn't retrigger this.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(()=>{saveSession(report,job)},[report,job?.id,job?.owner_token]);
  async function scan(e:ChangeEvent<HTMLInputElement>){
    const files=[...(e.target.files||[])].filter(f=>IMAGE_TYPES.has(f.type)); if(!files.length){setError("Choose a folder containing JPG, PNG, WebP, GIF, or BMP images.");return} setSelectedFiles(files);setJob(null);
    setError("");setPhase("scan");setProgress(1);const start=performance.now();
    try{
      let hashed=0;
      const items=await mapWithConcurrency(files,8,async(f)=>{
        const p=parsePath(f);const [hash,v]=await Promise.all([sha256(f),visualHash(f)]);
        hashed++;setProgress(Math.round((hashed/files.length)*72));
        return {name:f.name,path:p.path,split:p.split,label:p.label,size:f.size,hash,visual:v.hash,width:v.width,height:v.height};
      });
      const findings:Finding[]=[];const exactGroups=new Map<string,Item[]>();items.forEach(x=>exactGroups.set(x.hash,[...(exactGroups.get(x.hash)||[]),x]));
      const hasSplits=items.some(x=>x.split!=="unsplit");
      for(const group of exactGroups.values())for(let i=0;i<group.length;i++)for(let j=i+1;j<group.length;j++){const a=group[i],b=group[j];if(!hasSplits||a.split!==b.split){findings.push({kind:a.label!==b.label?"Label conflict":"Exact duplicate",severity:"Critical",a,b,similarity:100})}}
      setProgress(78);for(const [i,j] of nearCandidates(items)){const a=items[i],b=items[j];if((hasSplits&&a.split===b.split)||a.hash===b.hash)continue;const d=distance(a.visual,b.visual);if(d<=4)findings.push({kind:a.label!==b.label?"Label conflict":"Near duplicate",severity:"High",a,b,similarity:Math.round((1-d/64)*1000)/10});}
      const unique=new Map<string,Finding>();findings.forEach(f=>unique.set([f.kind,f.a.path,f.b.path].join("|"),f));const final=[...unique.values()].sort((a,b)=>b.similarity-a.similarity);
      const splits:Record<string,number>={},classes:Record<string,number>={};items.forEach(x=>{splits[x.split]=(splits[x.split]||0)+1;classes[x.label]=(classes[x.label]||0)+1});
      const exact=final.filter(f=>f.kind==="Exact duplicate").length,near=final.filter(f=>f.kind==="Near duplicate").length,conflicts=final.filter(f=>f.kind==="Label conflict").length;const affected=new Set(final.flatMap(f=>[f.a.path,f.b.path])).size;const score=Math.max(0,Math.round(100-(affected/items.length)*100*2-(conflicts/items.length)*100*3));
      const completed={name:parsePath(files[0]).root,items,findings:final,splits,classes,exact,near,conflicts,score,duration:performance.now()-start};setProgress(100);setReport(completed);setPhase("done");
    }catch(err){setError(err instanceof Error?err.message:"The scan could not be completed.");setPhase("idle")}
  }
  function choose(){if(input.current){input.current.setAttribute("webkitdirectory","");input.current.click()}}
  function ownerHeaders(current:ExperimentJob|null):Record<string,string>{return current?.owner_token?{[OWNER_HEADER]:current.owner_token}:{}}
  function runExperiment(){
    if(!report)return;
    if(!selectedFiles.length){setError("This tab no longer has your dataset files loaded. Choose the folder again with “New audit” to re-run the experiment.");return}
    const splitNames=Object.keys(report.splits);
    if(splitNames.some(s=>s!=="unsplit")&&!report.splits.train){setError(`This dataset has split folders (${splitNames.join(", ")}) but no train folder, so the benchmark cannot run. Choose the parent folder that also contains train/.`);return}
    setError("");setJob({id:"",status:"uploading",message:"Uploading the dataset to the local experiment worker",progress:1});
    const body=new FormData();body.append("dataset_name",report.name);body.append("epochs","5");selectedFiles.forEach(file=>{body.append("files",file,file.name);body.append("paths",(file as File&{webkitRelativePath?:string}).webkitRelativePath||file.name)});
    const request=new XMLHttpRequest();request.open("POST",`${API_URL}/api/experiments`);if(API_KEY)request.setRequestHeader("X-SplitGuard-Api-Key",API_KEY);request.upload.onprogress=(event)=>{if(event.lengthComputable){const value=Math.max(1,Math.min(7,Math.round(event.loaded/event.total*7)));setJob({id:"",status:"uploading",message:`Uploading dataset · ${Math.round(event.loaded/event.total*100)}%`,progress:value})}};request.onload=()=>{try{const payload=JSON.parse(request.responseText);if(request.status<200||request.status>=300)throw new Error(payload.detail||request.responseText);setJob(payload)}catch(err){setJob({id:"",status:"failed",message:err instanceof Error?err.message:"Could not start the experiment",progress:0})}};request.onerror=()=>setJob({id:"",status:"failed",message:"The experiment worker could not be reached. Check its URL and CORS settings.",progress:0});request.send(body);
  }
  // Re-polling on every status/progress tick would restart the interval constantly;
  // id/status/owner_token are the only fields that should restart it.
  useEffect(()=>{
    if(!job?.id||["complete","failed","cancelled"].includes(job.status))return;
    let failures=0;
    const tick=async()=>{
      try{
        const response=await fetch(`${API_URL}/api/experiments/${job.id}`,{headers:ownerHeaders(job)});
        if(response.status===404||response.status===403){setJob(null);setError("That experiment is no longer available on the worker (it expired, was deleted, or the worker was reset).");return}
        if(!response.ok)throw new Error(`status ${response.status}`);
        const payload=await response.json();failures=0;setError("");
        setJob(prev=>prev?{...payload,owner_token:prev.owner_token}:payload);
      }catch{failures++;if(failures>=3)setError("Lost contact with the experiment worker. Still retrying…")}
    };
    void tick();
    const timer=window.setInterval(tick,2500);
    return()=>window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[job?.id,job?.status,job?.owner_token]);
  function download(){if(!report)return;const blob=new Blob([JSON.stringify(report,null,2)],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=`${report.name}-splitguard-report.json`;a.click();URL.revokeObjectURL(a.href)}
  async function cancelExperiment(){if(!job?.id)return;await fetch(`${API_URL}/api/experiments/${job.id}/cancel`,{method:"POST",headers:ownerHeaders(job)});setJob({...job,status:"cancelled",message:"Experiment cancelled",progress:0})}
  async function deleteExperiment(){if(!job?.id)return;await fetch(`${API_URL}/api/experiments/${job.id}`,{method:"DELETE",headers:ownerHeaders(job)});setJob(null)}
  const risk=report?(report.score>=90?"Few automated flags":report.score>=70?"Review needed":"Many flags detected"):"";
  const impact=job?.result||null;
  const experimentEstimate=report?estimateExperiment(report):"";
  return <main id="top">
    <nav className="topbar"><a className="logo" href="#top" onClick={()=>window.scrollTo({top:0,behavior:"smooth"})}><span>SG</span>SplitGuard</a><div className="toplinks"><a href="#top">Overview</a><a href="#findings">Audit checks</a></div></nav>
    {!report&&phase!=="scan"&&<section className="welcome"><div className="kicker"><i/> DATASET EVALUATION, VERIFIED</div><h1>Know if your model is good.<br/><em>Or your split is lying.</em></h1><p>Audit image classification datasets for cross split duplicates, transformed copies, and label conflicts, before you trust another accuracy score.</p><button className="cta" onClick={choose}>Choose dataset folder <b>↗</b></button><input ref={input} hidden type="file" multiple accept="image/*" onChange={scan}/><small>Expected structure: train/class/images</small>{error&&<div className="error">{error}</div>}<div className="trust"><span><b>01</b> Exact matching</span><span><b>02</b> Visual similarity</span><span><b>03</b> Label conflicts</span></div></section>}
    {phase==="scan"&&<section className="scanning"><div className="scanner" style={{background:`conic-gradient(var(--accent) 0 ${progress}%, #e6e6eb ${progress}% 100%)`}} aria-label={`Audit ${progress}% complete`}><span>{progress}%</span></div><p className="kicker">AUDIT IN PROGRESS</p><h1>Reading the evidence.</h1><p>Hashing image contents and comparing records across dataset splits. Nothing is uploaded during this browser audit.</p><div className="progress"><i style={{width:`${progress}%`}}/></div><small>{progress<73?"Building image fingerprints":progress<90?"Comparing indexed similarity candidates":"Preparing integrity report"}</small></section>}
    {report&&<div className="app"><aside><div><p className="side-label">CURRENT AUDIT</p><strong>{report.name}</strong><small>{report.items.length.toLocaleString()} images · {Object.keys(report.classes).length} classes</small></div><button className="new" onClick={choose}>＋ New audit</button><input ref={input} hidden type="file" multiple accept="image/*" onChange={scan}/><div className="side-nav"><button className="active" onClick={()=>window.scrollTo({top:0,behavior:"smooth"})}>Overview</button><button onClick={()=>document.getElementById("findings")?.scrollIntoView({behavior:"smooth",block:"start"})}>Findings <b>{report.findings.length}</b></button><button onClick={download}>Export report</button></div><div className="local-note"><i/>Local analysis<br/><span>No image left your browser</span></div></aside>
      <section className="dashboard"><header><div><p className="kicker"><i/> AUDIT COMPLETE · {(report.duration/1000).toFixed(1)}S</p><h1>Integrity report</h1><p>Evidence generated from <b>{report.name}</b>, not sample data. This is a preliminary local scan; if you continue to training, the worker recomputes leakage from the uploaded files before deciding what to remove.</p></div><button className="export" onClick={download}>Download JSON report</button></header>
        <div className="scoreboard"><article className="score"><div className={`score-number ${report.score<70?"danger":""}`}>{report.score}</div><div><span>REVIEW SIGNAL / 100</span><h2>{risk}</h2><p>{report.findings.length?`${report.findings.length} relationships need review.`:"No duplicate relationships were detected."}</p><small>Heuristic screening aid, not a validated dataset-quality score.</small></div></article><article><span>EXACT LEAKAGE</span><strong>{report.exact}</strong><p>identical pairs</p></article><article><span>NEAR DUPLICATES</span><strong>{report.near}</strong><p>visual matches to review</p></article><article><span>LABEL CONFLICTS</span><strong>{report.conflicts}</strong><p>same content, different class</p></article></div>
        {!job&&<section className="next-step" aria-labelledby="next-step-title"><div className="next-step-number">NEXT</div><div><span>AUDIT FINISHED, CONTINUE HERE</span><h2 id="next-step-title">Train and validate the cleaned dataset</h2><p>Compare the original accuracy with the cleaned result. Estimated time for this dataset: <b>{experimentEstimate}</b>.</p></div><button onClick={()=>document.querySelector(".experiment")?.scrollIntoView({behavior:"smooth",block:"center"})}>Continue to training <b>↓</b></button></section>}
        <section className="grid"><article className="panel"><div className="panel-head"><div><span>DATASET COMPOSITION</span><h2>Split balance</h2></div><b>{report.items.length.toLocaleString()} total</b></div><div className="bars">{Object.entries(report.splits).sort((a,b)=>b[1]-a[1]).map(([name,count],i)=><div key={name}><label><span>{name}</span><b>{count.toLocaleString()}</b></label><div><i className={`bar c${i}`} style={{width:`${count/report.items.length*100}%`}}/></div><small>{(count/report.items.length*100).toFixed(1)}%</small></div>)}</div></article>
          <article className="panel experiment"><span>{impact?"CONTROLLED BENCHMARK RESULT":"TRAINING AND VALIDATION"}</span>{error&&<p className="error" style={{color:"#ff9a8f"}}>{error}</p>}{impact?<><div className="impact-values"><b>{(impact.baseline.test.accuracy*100).toFixed(2)}%</b><i>→</i><b>{(impact.cleaned.test.accuracy*100).toFixed(2)}%</b></div><h2>Accuracy changed by {impact.observed_difference_percentage_points>0?"+":""}{impact.observed_difference_percentage_points.toFixed(2)} points.</h2><p>MobileNetV2 benchmark, three-seed mean, same held out test set. Confidence range: ±{(impact.baseline.test.accuracy_ci95*100).toFixed(2)} to ±{(impact.cleaned.test.accuracy_ci95*100).toFixed(2)} points. Macro F1: {(impact.baseline.test.macro_f1*100).toFixed(2)}% to {(impact.cleaned.test.macro_f1*100).toFixed(2)}%. {job?.preparation?`${job.preparation.removed_training_images} leaking or conflicting training images removed.`:""}</p><button onClick={deleteExperiment}>Delete experiment data</button></>:job?<><div className="lock">{job.status==="failed"?"!":"…"}</div><h2>{job.status==="failed"?"Experiment stopped":`${job.status[0].toUpperCase()}${job.status.slice(1)}`}</h2><p>{job.message}</p><div className="job-progress-meta"><strong>{Math.round(job.progress)}%</strong><span>{job.status==="training"?"Training progress":"Experiment progress"}</span></div><div className="job-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(job.progress)}><i style={{width:`${job.progress}%`}}/></div>{job.status==="failed"?<button onClick={runExperiment}>Try again</button>:job.status!=="cancelled"&&<button onClick={cancelExperiment}>Cancel</button>}</>:<><div className="lock">02</div><h2>Run the accuracy comparison</h2><p>Estimated time: <b>{experimentEstimate}</b>. SplitGuard will train the same benchmark on the original and cleaned dataset, then show both measured accuracies. Actual time depends on your hardware.</p><button className="run-benchmark" onClick={runExperiment}>I agree, upload and start training</button></>}</article></section>
        <section className="panel findings" id="findings"><div className="panel-head"><div><span>AUDIT EVIDENCE</span><h2>Findings across dataset splits</h2></div><div className="filters">{["All","Exact duplicate","Near duplicate","Label conflict"].map(x=><button key={x} className={filter===x?"active":""} onClick={()=>setFilter(x)}>{x}</button>)}</div></div>{visible.length===0?<div className="clear"><b>✓</b><h3>No findings in this category</h3><p>No matching pair was detected across the dataset splits.</p></div>:<div className="table"><div className="tr heading"><span>Risk</span><span>Source</span><span>Match</span><span>Similarity</span></div>{visible.slice(0,100).map((f,i)=><div className="tr" key={`${f.a.path}${f.b.path}${i}`}><span><i className={`badge ${f.kind==="Label conflict"?"red":"orange"}`}>{f.kind}</i></span><span><b>{f.a.name}</b><small>{f.a.split} / {f.a.label}</small></span><span><b>{f.b.name}</b><small>{f.b.split} / {f.b.label}</small></span><span className="similarity">{f.similarity}%</span></div>)}</div>}</section>
        <footer>Results should be reviewed before removing files.</footer>
      </section></div>}
  </main>
}
