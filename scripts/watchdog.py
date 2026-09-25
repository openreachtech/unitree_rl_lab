import re,sys,os,time,subprocess
log,ckpt_iter,rules = sys.argv[1],int(sys.argv[2]),sys.argv[3]
# rules: "metric:op:value,..."  op は lt/gt
R=[]
for r in rules.split(","):
    k,op,v=r.split(":"); R.append((k,op,float(v)))
def vals(blk):
    d={}
    for k,_,_ in R:
        m=re.search(re.escape(k)+r":\s*([\-\d.]+)",blk)
        if m: d[k]=float(m.group(1))
    return d
t0=time.time()
while time.time()-t0 < 4200:
    time.sleep(30)
    try: txt=open(log,errors="ignore").read()
    except Exception: continue
    b=re.split(r"Learning iteration (\d+)/",txt)
    if len(b)<3: continue
    last_it=int(b[-2]); d=vals(b[-1])
    if "exited with code" in txt:
        print("完了"); break
    if last_it>=ckpt_iter:
        bad=[f"{k}={d[k]:.3f}(要 {op} {v})" for k,op,v in R if k in d and ((op=="lt" and d[k]>=v) or (op=="gt" and d[k]<=v))]
        if bad:
            pid=subprocess.run(["pgrep","-f","max_iterations 1500 --resume"],capture_output=True,text=True).stdout.split()
            for p in pid: subprocess.run(["kill","-9",p])
            print(f"中止条件に抵触 iter{last_it}: "+" / ".join(bad)); break
