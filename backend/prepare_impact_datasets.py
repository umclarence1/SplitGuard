from __future__ import annotations
import argparse, hashlib, json, random, shutil
from collections import Counter, defaultdict
from pathlib import Path

def copy_unique(source:Path,destination:Path,relative:str)->None:
    token=hashlib.sha1(relative.encode()).hexdigest()[:10]; target=destination/f"{token}-{source.name}";target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)

def stratified(rows:list[dict],seed:int)->tuple[dict[str,list[dict]],list[dict]]:
    rng=random.Random(seed);result={"train":[],"validation":[],"test":[]};by_label=defaultdict(list);dropped=[]
    for r in rows:by_label[r["label"]].append(r)
    for label,items in sorted(by_label.items()):
        rng.shuffle(items);n=len(items)
        if n<3:dropped.append({"label":label,"count":n});continue
        test=max(1,round(n*.15));val=max(1,round(n*.15));test=min(test,n-2);val=min(val,n-test-1)
        result["test"]+=items[:test];result["validation"]+=items[test:test+val];result["train"]+=items[test+val:]
    return result,dropped

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--audit",type=Path,required=True);p.add_argument("--dataset",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",type=int,default=42);a=p.parse_args()
    inventory=json.loads((a.audit/"inventory.json").read_text(encoding="utf-8"));rels=json.loads((a.audit/"relationships.json").read_text(encoding="utf-8"));valid=[r for r in inventory if "error" not in r]
    supplied=any(r["split"]!="unsplit" for r in valid);dropped_classes=[]
    if supplied:
        splits={s:[r for r in valid if r["split"]==s] for s in ("train","validation","test")}
        if not splits["train"]:raise ValueError("A supplied split dataset must contain a train folder.")
        needs_validation=False
        if not splits["test"] and splits["validation"]:
            splits["test"]=splits["validation"]
            splits["validation"]=[]
            needs_validation=True
        elif not splits["test"]:
            splits,dropped_classes=stratified(splits["train"],a.seed)
        if not splits["validation"] or needs_validation:
            rng=random.Random(a.seed);remaining=[];validation=[];by_label=defaultdict(list)
            for row in splits["train"]:by_label[row["label"]].append(row)
            for rows in by_label.values():
                rng.shuffle(rows);take=max(1,round(len(rows)*.15)) if len(rows)>1 else 0;validation+=rows[:take];remaining+=rows[take:]
            splits["train"]=remaining;splits["validation"]=validation
    else:splits,dropped_classes=stratified(valid,a.seed)
    assigned={r["relative_path"]:s for s,rows in splits.items() for r in rows};baseline_train=list(splits["train"]);remove=set();reasons=defaultdict(set)
    exact_test_conflicts=[];review_test_relationships=[];review_validation_relationships=[]
    for rel in rels:
        x,y=rel["a"],rel["b"];sx,sy=assigned.get(x),assigned.get(y)
        if sx is None or sy is None:continue
        if sx=="test" and sy=="test":
            if rel["label_conflict"]:
                entry={"a":x,"b":y,"kind":rel["kind"]}
                (exact_test_conflicts if rel["kind"]=="exact" else review_test_relationships).append(entry)
            continue
        if sx=="train" and sy in ("validation","test"):remove.add(x);reasons[x].add("matches held-out evaluation image")
        if sy=="train" and sx in ("validation","test"):remove.add(y);reasons[y].add("matches held-out evaluation image")
        if sx=="validation" and sy=="test":remove.add(x);reasons[x].add("matches test image; removed from validation to protect evaluation integrity")
        if sy=="validation" and sx=="test":remove.add(y);reasons[y].add("matches test image; removed from validation to protect evaluation integrity")
        if rel["label_conflict"]:
            if sx=="train":remove.add(x);reasons[x].add("cross-label conflict")
            if sy=="train":remove.add(y);reasons[y].add("cross-label conflict")
            if sx=="validation" and sy=="validation":review_validation_relationships.append({"a":x,"b":y,"kind":rel["kind"]})
    if exact_test_conflicts:
        pairs=", ".join(f"{e['a']} vs {e['b']}" for e in exact_test_conflicts[:5])
        raise ValueError(f"{len(exact_test_conflicts)} byte-identical test image pair(s) carry conflicting labels ({pairs}); fix the test set labels before benchmarking against it.")
    cleaned_train=[r for r in baseline_train if r["relative_path"] not in remove]
    validation=[r for r in splits["validation"] if r["relative_path"] not in remove]
    eligible={r["label"] for r in cleaned_train}&{r["label"] for r in splits["test"]}
    if len(eligible)<2:raise ValueError("Cleaning left fewer than two trainable classes with test examples.")
    baseline_train=[r for r in baseline_train if r["label"] in eligible];cleaned_train=[r for r in cleaned_train if r["label"] in eligible];validation=[r for r in validation if r["label"] in eligible];test=[r for r in splits["test"] if r["label"] in eligible]
    roots={"baseline":a.output/"baseline_original","cleaned":a.output/"cleaned"}
    if a.output.exists():shutil.rmtree(a.output)
    for variant,root in roots.items():
        chosen={"train":baseline_train if variant=="baseline" else cleaned_train,"val":validation,"test":test}
        for split,rows in chosen.items():
            for r in rows:copy_unique(a.dataset/r["relative_path"],root/split/r["label"],r["relative_path"])
    removed=[{"path":path,"reasons":sorted(reasons[path])} for path in sorted(remove)]
    summary={"policy":"Preserve one held-out test set; remove from cleaned training every exact/near match to held-out data and every cross-label training conflict; drop validation images that match a test image.","source_split":"supplied" if supplied else "generated","seed":a.seed,"raw_images":len(valid),"eligible_classes":len(eligible),"removed_training_images":len(remove),"baseline":{"train":len(baseline_train),"val":len(validation),"test":len(test)},"cleaned":{"train":len(cleaned_train),"val":len(validation),"test":len(test)},"same_validation_set":True,"same_test_set":True,"class_counts_cleaned_train":dict(Counter(r["label"] for r in cleaned_train)),"dropped_classes_insufficient_examples":dropped_classes,"unresolved_relationships_within_test":review_test_relationships,"unresolved_relationships_within_validation":review_validation_relationships}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/"preparation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");(a.output/"removed_training_images.json").write_text(json.dumps(removed,indent=2),encoding="utf-8");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
