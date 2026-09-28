from __future__ import annotations
import argparse,json,random,time
from copy import deepcopy
from pathlib import Path
import numpy as np,torch,torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score,f1_score
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights,mobilenet_v2

EXT={".jpg",".jpeg",".png",".bmp",".gif",".tif",".tiff",".webp"}
def seed_all(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
class FixedFolder(Dataset):
    def __init__(self,root:Path,class_map:dict[str,int],tf):
        self.samples=[(p,class_map[p.parent.name]) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in EXT and p.parent.name in class_map];self.tf=tf
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        path,label=self.samples[i]
        with Image.open(path) as im:image=im.convert("RGB")
        return self.tf(image),label
def loaders(root:Path,size:int,batch:int,seed:int):
    classes=sorted(p.name for p in (root/"train").iterdir() if p.is_dir());mapping={x:i for i,x in enumerate(classes)};mean,std=[.485,.456,.406],[.229,.224,.225]
    train_tf=transforms.Compose([transforms.Resize((size,size)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)]);eval_tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize(mean,std)])
    sets={s:FixedFolder(root/s,mapping,train_tf if s=="train" else eval_tf) for s in ("train","val","test")}
    if any(len(ds)==0 for ds in sets.values()):raise ValueError("Train, validation and test sets must each contain readable images.")
    return {s:DataLoader(ds,batch_size=batch,shuffle=s=="train",num_workers=0,generator=torch.Generator().manual_seed(seed)) for s,ds in sets.items()},classes
def epoch(model,loader,loss_fn,device,opt=None):
    model.train(opt is not None);total=correct=0;loss_sum=0.;ys=[];ps=[]
    for images,labels in loader:
        images,labels=images.to(device),labels.to(device)
        if opt:opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(opt is not None):out=model(images);loss=loss_fn(out,labels);loss.backward() if opt else None;opt.step() if opt else None
        pred=out.argmax(1);total+=labels.numel();correct+=(pred==labels).sum().item();loss_sum+=loss.item()*labels.numel();ys+=labels.cpu().tolist();ps+=pred.cpu().tolist()
    return {"loss":loss_sum/total,"accuracy":correct/total,"macro_f1":f1_score(ys,ps,average="macro",zero_division=0),"balanced_accuracy":balanced_accuracy_score(ys,ps)}
def run_once(root:Path,args,device,seed:int,variant:str,seed_index:int):
    seed_all(seed);data,classes=loaders(root,args.image_size,args.batch_size,seed);model=mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    for p in model.features.parameters():p.requires_grad=False
    model.classifier[1]=nn.Linear(model.classifier[1].in_features,len(classes));model.to(device);loss_fn=nn.CrossEntropyLoss();opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=args.learning_rate);best=-1.;state=deepcopy(model.state_dict());history=[]
    for number in range(1,args.epochs+1):
        train=epoch(model,data["train"],loss_fn,device,opt);val=epoch(model,data["val"],loss_fn,device);history.append({"epoch":number,"train":train,"validation":val});print(f"{variant} seed {seed_index}/{len(args.seeds)} epoch {number}/{args.epochs} train={train['accuracy']:.4f} val={val['accuracy']:.4f}",flush=True)
        if val["accuracy"]>best:best=val["accuracy"];state=deepcopy(model.state_dict())
    model.load_state_dict(state);return {"seed":seed,"best_validation_accuracy":best,"test":epoch(model,data["test"],loss_fn,device),"history":history}
def aggregate(root:Path,args,device,variant:str):
    started=time.time();runs=[run_once(root,args,device,seed,variant,i+1) for i,seed in enumerate(args.seeds)];acc=np.array([r["test"]["accuracy"] for r in runs]);f1=np.array([r["test"]["macro_f1"] for r in runs]);ci=1.96*acc.std(ddof=1)/np.sqrt(len(acc)) if len(acc)>1 else 0
    return {"dataset":str(root),"model":"mobilenet_v2_imagenet_linear_probe","runs":runs,"test":{"accuracy":float(acc.mean()),"accuracy_std":float(acc.std(ddof=1)) if len(acc)>1 else 0.0,"accuracy_ci95":float(ci),"macro_f1":float(f1.mean())},"seconds":round(time.time()-started,2)}
def main():
    p=argparse.ArgumentParser();p.add_argument("--baseline",type=Path,required=True);p.add_argument("--cleaned",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--epochs",type=int,default=5);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--image-size",type=int,default=160);p.add_argument("--learning-rate",type=float,default=3e-4);p.add_argument("--seeds",type=int,nargs="+",default=[42,1337,2027]);a=p.parse_args();device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result={"protocol":{"benchmark_not_user_model":True,"same_model":True,"same_validation_set":True,"same_test_set":True,"seeds":a.seeds,"device":str(device)},"baseline":aggregate(a.baseline,a,device,"baseline",),"cleaned":aggregate(a.cleaned,a,device,"cleaned")};b=result["baseline"]["test"]["accuracy"]*100;c=result["cleaned"]["test"]["accuracy"]*100;result["observed_difference_percentage_points"]=round(c-b,2)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/"impact_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
