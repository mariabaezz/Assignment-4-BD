import csv, random, subprocess, time, statistics, re, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
SIZES=[500,1000,1500,2000,3000]
WORKERS=[1,2,4,8,10]
CONFIGS=[('seq',1)]+[(m,p) for m in ('omp','mpi') for p in WORKERS]
FIELDS=['n','method','workers','repetition','compute_s','operation_s','end_to_end_s','rss_max_bytes','rss_sum_bytes','checksum','setup_s','bcast_s','scatter_s','gather_s','comm_s','status']
def run(n,m,p,rep):
    cmd=['mpirun','-np',str(p),'./bin/mpi_matmul',str(n)] if m=='mpi' else ['./bin/run_parallel',m,str(n),str(p)]
    t=time.perf_counter()
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=120,check=True)
    elapsed=time.perf_counter()-t
    d=dict(re.findall(r'(\w+)=([^\s,]+)',r.stdout.strip().splitlines()[-1]))
    row=dict(n=n,method=m,workers=p,repetition=rep,compute_s=float(d['compute' if m=='mpi' else 'time']),operation_s=float(d['total' if m=='mpi' else 'time']),end_to_end_s=elapsed,rss_max_bytes=int(d['rss_max_bytes']),rss_sum_bytes=int(d.get('rss_sum_bytes',d['rss_max_bytes'])),checksum=int(d['checksum']),status='ok')
    for k in ('setup','bcast','scatter','gather','comm'):
        if k in d: row[k+'_s']=float(d[k])
    return row
if __name__=='__main__':
    for n in (1,3,7,11):
        checks={run(n,m,p,0)['checksum'] for m,p in [('seq',1),('omp',4),('mpi',4)]}
        assert len(checks)==1,(n,checks)
    print('Small matrix checksum tests passed',flush=True)
    rng=random.Random(42)
    with open('results/results_final.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
        for n in SIZES:
            configs=CONFIGS.copy();rng.shuffle(configs)
            failed=set(); checks=set()
            for rep in range(6):
                rng.shuffle(configs)
                for m,p in configs:
                    if (m,p) in failed: continue
                    try: row=run(n,m,p,rep)
                    except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as e:
                        failed.add((m,p));row=dict(n=n,method=m,workers=p,repetition=rep,status='timeout' if isinstance(e,subprocess.TimeoutExpired) else 'error');w.writerow(row);f.flush();print(e,flush=True);continue
                    checks.add(row['checksum']);assert len(checks)==1,(n,checks)
                    if rep: w.writerow(row);f.flush()
                    print(n,m,p,rep,row['end_to_end_s'],flush=True)
