from __future__ import annotations
import json,os,queue,re,secrets,shutil,subprocess,sys,threading,time,uuid
from contextlib import asynccontextmanager
from pathlib import Path,PurePosixPath
from fastapi import FastAPI,HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile

ROOT=Path(__file__).resolve().parents[1];JOBS=ROOT/"jobs";PYTHON=Path(sys.executable);MAX_FILES=int(os.getenv("SPLITGUARD_MAX_FILES","25000"));MAX_TOTAL=int(os.getenv("SPLITGUARD_MAX_UPLOAD_BYTES",str(20*1024**3)));RETENTION=int(os.getenv("SPLITGUARD_RETENTION_HOURS","24"));WORKERS=max(1,int(os.getenv("SPLITGUARD_WORKERS","1")))
TRAIN_BATCH_SIZE=int(os.getenv("SPLITGUARD_TRAIN_BATCH_SIZE","32"));TRAIN_IMAGE_SIZE=int(os.getenv("SPLITGUARD_TRAIN_IMAGE_SIZE","160"))
IMAGE_EXTENSIONS={".png",".jpg",".jpeg",".bmp",".gif",".tif",".tiff",".webp"}
API_KEY=os.getenv("SPLITGUARD_API_KEY","").strip()
def parse_origins(raw:str)->list[str]:return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
ORIGINS=parse_origins(os.getenv("SPLITGUARD_CORS_ORIGINS","http://localhost:5173,http://127.0.0.1:5173"));work:queue.Queue[tuple[str,int]|None]=queue.Queue();active:dict[str,subprocess.Popen]={};lock=threading.Lock()
app=FastAPI(title="SplitGuard Worker",version="0.4.0")
app.add_middleware(CORSMiddleware,allow_origins=ORIGINS,allow_credentials=False,allow_methods=["GET","POST","DELETE","OPTIONS"],allow_headers=["Content-Type","X-SplitGuard-Api-Key","X-SplitGuard-Owner-Token"])
def state_path(job:str)->Path:return JOBS/job/"state.json"
def read_state(job:str):return json.loads(state_path(job).read_text(encoding="utf-8"))
def write_state(job:str,**values):
    with lock:
        path=state_path(job);current=read_state(job) if path.exists() else {"id":job,"created_at":time.time()};current.update(values);tmp=path.with_suffix(".tmp");tmp.parent.mkdir(parents=True,exist_ok=True);tmp.write_text(json.dumps(current,indent=2),encoding="utf-8");tmp.replace(path)
def cancelled(job:str)->bool:return state_path(job).exists() and read_state(job).get("status")=="cancelled"
def public(state:dict)->dict:return {k:v for k,v in state.items() if k!="owner_token"}
def require_api_key(request:Request)->None:
    if API_KEY and request.headers.get("x-splitguard-api-key","")!=API_KEY:raise HTTPException(401,"Missing or invalid API key.")
def require_owner(job:str,request:Request)->dict:
    state=read_state(job);token=state.get("owner_token")
    if token and request.headers.get("x-splitguard-owner-token","")!=token:raise HTTPException(403,"Missing or invalid owner token for this experiment.")
    return state
def stream(job:str,cmd:list[str],phase:str):
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with lock:active[job]=process
    output=[]
    try:
        assert process.stdout
        for line in process.stdout:
            output.append(line)
            if cancelled(job):
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise RuntimeError("Experiment cancelled")
            if phase=="audit":
                m=re.search(r"fingerprinted (\d+)/(\d+)",line)
                if m:
                    done,total=map(int,m.groups());pct=8+round(done/total*24);write_state(job,status="auditing",message=f"Fingerprinting images · {done:,} of {total:,}",progress=pct)
            else:
                m=re.search(r"(baseline|cleaned) seed (\d+)/(\d+) epoch (\d+)/(\d+)",line)
                if m:
                    groups=m.groups();variant=groups[0];seed,total_seeds,epoch,total_epochs=map(int,groups[1:]);fraction=((seed-1)*total_epochs+epoch)/(total_seeds*total_epochs);base,width=(48,22) if variant=="baseline" else (72,22);pct=base+round(fraction*width);write_state(job,status="training",message=f"Training {variant} benchmark · seed {seed}/{total_seeds} · epoch {epoch}/{total_epochs}",progress=pct)
        if process.wait():raise RuntimeError("".join(output[-40:]))
    finally:
        with lock:active.pop(job,None)
def run_job(job:str,epochs:int):
    folder=JOBS/job;raw=folder/"raw";audit=folder/"audit";prepared=folder/"prepared";result=folder/"result"
    try:
        write_state(job,status="auditing",message="Computing exact and visual fingerprints",progress=8);stream(job,[str(PYTHON),str(ROOT/"backend/audit_dataset.py"),str(raw),"--output",str(audit)],"audit")
        if cancelled(job):return
        write_state(job,status="cleaning",message="Removing training leakage while preserving the held-out test set",progress=35,audit=json.loads((audit/"summary.json").read_text(encoding="utf-8")))
        subprocess.run([str(PYTHON),str(ROOT/"backend/prepare_impact_datasets.py"),"--audit",str(audit),"--dataset",str(raw),"--output",str(prepared)],check=True,capture_output=True,text=True)
        if cancelled(job):return
        write_state(job,status="training",message="Running three-seed controlled benchmark",progress=48);stream(job,[str(PYTHON),str(ROOT/"backend/run_impact_experiment.py"),"--baseline",str(prepared/"baseline_original"),"--cleaned",str(prepared/"cleaned"),"--output",str(result),"--epochs",str(epochs),"--batch-size",str(TRAIN_BATCH_SIZE),"--image-size",str(TRAIN_IMAGE_SIZE)],"training")
        results=json.loads((result/"impact_results.json").read_text(encoding="utf-8"));write_state(job,status="complete",message="Measured benchmark accuracies are ready",progress=100,result=results,preparation=json.loads((prepared/"preparation_summary.json").read_text(encoding="utf-8")),completed_at=time.time())
    except subprocess.CalledProcessError as exc:
        raw=(exc.stderr or exc.stdout or str(exc)).strip().splitlines();message=raw[-1] if raw else "Dataset preparation failed"
        if "ValueError:" in message:message=message.split("ValueError:",1)[1].strip()
        write_state(job,status="failed",message=message,progress=0)
    except Exception as exc:
        if not cancelled(job):write_state(job,status="failed",message=str(exc)[-2000:],progress=0)
def worker():
    while True:
        item=work.get()
        if item is None:return
        job,epochs=item
        if not cancelled(job):run_job(job,epochs)
        work.task_done()
def cleanup():
    cutoff=time.time()-RETENTION*3600
    for path in JOBS.glob("*/state.json") if JOBS.exists() else []:
        try:
            state=json.loads(path.read_text(encoding="utf-8"));stamp=state.get("completed_at",state.get("created_at",time.time()))
            if state.get("status") in {"complete","failed","cancelled"} and stamp<cutoff:shutil.rmtree(path.parent)
        except Exception:pass
def cleanup_loop():
    while True:
        cleanup();time.sleep(3600)
@asynccontextmanager
async def lifespan(_app:FastAPI):
    JOBS.mkdir(exist_ok=True)
    threading.Thread(target=cleanup_loop,daemon=True).start()
    for _ in range(WORKERS):threading.Thread(target=worker,daemon=True).start()
    for path in JOBS.glob("*/state.json"):
        state=json.loads(path.read_text(encoding="utf-8"))
        if state.get("status") in {"queued","auditing","cleaning","training"}:write_state(state["id"],status="queued",message="Recovered job queued",progress=5);work.put((state["id"],int(state.get("epochs",5))))
    yield
app.router.lifespan_context=lifespan
@app.get("/api/health")
def health():return {"status":"ok","workers":WORKERS,"queued":work.qsize(),"retention_hours":RETENTION}
def _cancel(job:str)->None:
    write_state(job,status="cancelled",message="Experiment cancelled",progress=0,completed_at=time.time())
    with lock:
        if job in active:active[job].terminate()
@app.post("/api/experiments")
async def create(request:Request):
    require_api_key(request)
    form=await request.form(max_files=MAX_FILES,max_fields=MAX_FILES,max_part_size=100*1024*1024);files=[x for x in form.getlist("files") if isinstance(x,UploadFile)];paths=[str(x) for x in form.getlist("paths")]
    if not files or len(files)!=len(paths):raise HTTPException(400,"Every uploaded image requires a relative path.")
    if len(files)>MAX_FILES:raise HTTPException(413,"Dataset has too many files for this worker.")
    try:epochs=max(1,min(int(str(form.get("epochs","5"))),30))
    except ValueError:raise HTTPException(400,"epochs must be an integer")
    job=uuid.uuid4().hex[:12];token=secrets.token_urlsafe(24);raw=JOBS/job/"raw";raw.mkdir(parents=True);write_state(job,status="uploading",message="Saving dataset",progress=2,dataset_name=str(form.get("dataset_name","dataset"))[:200],file_count=len(files),epochs=epochs,owner_token=token);total=0;saved=0
    try:
        for upload,relative in zip(files,paths):
            safe=PurePosixPath(relative.replace("\\","/"));parts=[p for p in safe.parts if p not in ("",".","..")];parts=parts[1:] if len(parts)>1 else parts
            if not parts or Path(parts[-1]).suffix.lower() not in IMAGE_EXTENSIONS:
                await upload.close();continue
            destination=raw.joinpath(*parts);destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open("wb") as handle:
                while chunk:=await upload.read(1024*1024):
                    total+=len(chunk)
                    if total>MAX_TOTAL:raise HTTPException(413,"Dataset exceeds this worker's upload limit.")
                    handle.write(chunk)
            await upload.close();saved+=1
    except Exception:shutil.rmtree(JOBS/job,ignore_errors=True);raise
    if not saved:shutil.rmtree(JOBS/job,ignore_errors=True);raise HTTPException(400,"No readable image files were found in the upload.")
    write_state(job,status="queued",message=f"Experiment queued · {work.qsize()+1} ahead or running",progress=5,upload_bytes=total);work.put((job,epochs));return {**public(read_state(job)),"owner_token":token}
@app.get("/api/experiments/{job}")
def get(job:str,request:Request):
    if not state_path(job).exists():raise HTTPException(404,"Experiment not found")
    return public(require_owner(job,request))
@app.post("/api/experiments/{job}/cancel")
def cancel(job:str,request:Request):
    if not state_path(job).exists():raise HTTPException(404,"Experiment not found")
    require_owner(job,request);_cancel(job)
    return public(read_state(job))
@app.delete("/api/experiments/{job}")
def delete(job:str,request:Request):
    if not state_path(job).exists():raise HTTPException(404,"Experiment not found")
    require_owner(job,request);_cancel(job);shutil.rmtree(JOBS/job,ignore_errors=True)
    return {"deleted":True}
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)
