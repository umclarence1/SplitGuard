from __future__ import annotations

import argparse, hashlib, json, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

EXTENSIONS={".png",".jpg",".jpeg",".bmp",".gif",".tif",".tiff",".webp"}
SPLITS={"train":"train","training":"train","val":"validation","valid":"validation","validation":"validation","test":"test","testing":"test"}

def file_hash(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def dhash(path:Path,size:int=8)->int:
    with Image.open(path) as src:
        im=ImageOps.exif_transpose(src).convert("L").resize((size+1,size),Image.Resampling.LANCZOS)
        px=np.asarray(im,dtype=np.int16)
    bits=px[:,1:]>px[:,:-1]; value=0
    for bit in bits.flat:value=(value<<1)|int(bit)
    return value

def metadata(root:Path,path:Path)->tuple[str,str]:
    parts=path.relative_to(root).parts
    split_index=next((i for i,p in enumerate(parts) if p.lower() in SPLITS),-1)
    if split_index>=0:
        split=SPLITS[parts[split_index].lower()]
        label=parts[split_index+1] if split_index+1<len(parts) else "unlabeled"
    else:
        split="unsplit"; label=parts[-2] if len(parts)>1 else "unlabeled"
    return split,label

class BKNode:
    def __init__(self,value:int):self.value=value;self.children={}
    def add(self,value:int):
        node=self
        while True:
            d=(node.value^value).bit_count()
            if d==0:return
            if d not in node.children:node.children[d]=BKNode(value);return
            node=node.children[d]
    def search(self,value:int,radius:int):
        d=(self.value^value).bit_count()
        if d<=radius:yield self.value
        for edge,child in self.children.items():
            if d-radius<=edge<=d+radius:yield from child.search(value,radius)

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("dataset",type=Path);p.add_argument("--output",type=Path,required=True);p.add_argument("--near-distance",type=int,default=4);a=p.parse_args();started=time.time()
    paths=sorted(x for x in a.dataset.rglob("*") if x.is_file() and x.suffix.lower() in EXTENSIONS);records=[]
    for index,path in enumerate(paths,1):
        split,label=metadata(a.dataset,path)
        try:
            with Image.open(path) as im:width,height=im.size
            records.append({"path":str(path),"relative_path":path.relative_to(a.dataset).as_posix(),"split":split,"label":label,"sha256":file_hash(path),"dhash":dhash(path),"width":width,"height":height,"bytes":path.stat().st_size})
        except Exception as exc:records.append({"path":str(path),"relative_path":path.relative_to(a.dataset).as_posix(),"split":split,"label":label,"error":repr(exc)})
        if index%100==0 or index==len(paths):print(f"fingerprinted {index}/{len(paths)}",flush=True)
    valid=[r for r in records if "error" not in r];by_sha=defaultdict(list);by_dhash=defaultdict(list)
    for r in valid:by_sha[r["sha256"]].append(r);by_dhash[r["dhash"]].append(r)
    exact=[];relationships=[];exact_keys=set()
    for group in by_sha.values():
        if len(group)<2:continue
        conflict=len({r["label"] for r in group})>1;cross_split=len({r["split"] for r in group})>1 and "unsplit" not in {r["split"] for r in group}
        exact.append({"sha256":group[0]["sha256"],"images":[{"path":r["relative_path"],"label":r["label"],"split":r["split"]} for r in group],"label_conflict":conflict,"cross_split":cross_split})
        for i,x in enumerate(group):
            for y in group[i+1:]:
                key=tuple(sorted((x["relative_path"],y["relative_path"])));exact_keys.add(key)
                relationships.append({"kind":"exact","a":key[0],"b":key[1],"distance":0,"similarity":100.0,"label_conflict":x["label"]!=y["label"],"cross_split":x["split"]!=y["split"] and "unsplit" not in (x["split"],y["split"])})
    unique=list(by_dhash);tree=BKNode(unique[0]) if unique else None
    if tree:
        for value in unique[1:]:tree.add(value)
        seen=set()
        for value in unique:
            for other in tree.search(value,a.near_distance):
                if other<value:continue
                for x in by_dhash[value]:
                    for y in by_dhash[other]:
                        if x is y:continue
                        key=tuple(sorted((x["relative_path"],y["relative_path"])))
                        if key in exact_keys or key in seen:continue
                        seen.add(key);d=(value^other).bit_count()
                        relationships.append({"kind":"near","a":key[0],"b":key[1],"distance":d,"similarity":round((64-d)/64*100,2),"label_conflict":x["label"]!=y["label"],"cross_split":x["split"]!=y["split"] and "unsplit" not in (x["split"],y["split"])})
    lookup={r["relative_path"]:r for r in valid};near=[{**row,"a_label":lookup[row["a"]]["label"],"b_label":lookup[row["b"]]["label"],"a_split":lookup[row["a"]]["split"],"b_split":lookup[row["b"]]["split"]} for row in relationships if row["kind"]=="near"]
    affected={p for row in relationships for p in (row["a"],row["b"])};conflicts={p for row in relationships if row["label_conflict"] for p in (row["a"],row["b"])}
    split_counts=Counter(r["split"] for r in valid);class_counts=Counter(r["label"] for r in valid);class_split=defaultdict(Counter)
    for r in valid:class_split[r["label"]][r["split"]]+=1
    summary={"dataset":str(a.dataset),"images":len(paths),"readable_images":len(valid),"corrupt_images":len(paths)-len(valid),"classes":len(class_counts),"class_counts":dict(sorted(class_counts.items())),"split_counts":dict(split_counts),"class_split_counts":{k:dict(v) for k,v in sorted(class_split.items())},"unique_sha256":len(by_sha),"exact_duplicate_groups":len(exact),"exact_cross_split_groups":sum(x["cross_split"] for x in exact),"exact_cross_label_groups":sum(x["label_conflict"] for x in exact),"near_duplicate_pairs":len(near),"near_cross_split_pairs":sum(x["cross_split"] for x in near),"near_cross_label_pairs":sum(x["label_conflict"] for x in near),"affected_images":len(affected),"images_requiring_conflict_review":len(conflicts),"near_hash_distance":a.near_distance,"elapsed_seconds":round(time.time()-started,2),"method":"SHA-256 plus BK-tree indexed 64-bit dHash candidate search"}
    a.output.mkdir(parents=True,exist_ok=True)
    for name,data in (("summary.json",summary),("exact_groups.json",exact),("near_pairs.json",near),("relationships.json",relationships),("inventory.json",records)):(a.output/name).write_text(json.dumps(data,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
