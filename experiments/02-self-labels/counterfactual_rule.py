import json, glob, collections, random
fam=('rename','typos','distract')
data=collections.defaultdict(dict)
for f in glob.glob('runs/02-self-labels/*.jsonl'):
    m=f.split('/')[-1].replace('.jsonl','')
    for l in open(f):
        r=json.loads(l); data[(m,r['problem_id'])][r['family']]=(r['answers'],r['answer'])
def label(e, oi, vi):
    a,g=e['original']; base=sum(a[i]==g for i in oi)/len(oi)
    if base==0: return True
    if not all(f in e for f in fam): return None
    md=max(base-sum(e[f][0][i]==g for i in vi)/len(vi) for f in fam)
    return True if md<=0.10 else False if md>=0.25 else None
def feat(e, oi, vi):
    if not all(f in e for f in fam): return None
    a,_=e['original']; orig=[a[i] for i in oi]
    c=collections.Counter(x for x in orig if x is not None)
    if not c: return (0.0,0.0)
    maj,cnt=c.most_common(1)[0]; p0=cnt/len(orig)
    return (max(p0-sum(e[f][0][i]==maj for i in vi)/len(vi) for f in fam), p0)
rows=[]
for (m,p),e in data.items():
    if 'original' not in e: continue
    y=label(e, range(4,8), range(3,6))
    if y is None: continue
    rows.append((m,p,y,feat(e, range(0,4), range(0,3))))
per=collections.defaultdict(list)
for r in rows: per[r[1]].append(r)
mixed={p:v for p,v in per.items() if len({x[2] for x in v})==2}
print('labeled',len(rows),'mixed problems',len(mixed))
def pred(r,t):
    if r[3] is None: return True
    return r[3][0]<=t
rng=random.Random(0)
for t in (0.0,0.1,0.2,0.3,0.5):
    accs=[]
    for _ in range(2000):
        ok=n=0
        for p,v in mixed.items():
            pos=[x for x in v if x[2]]; neg=[x for x in v if not x[2]]
            for r in (rng.choice(pos), rng.choice(neg)):
                ok+=pred(r,t)==r[2]; n+=1
        accs.append(ok/n)
    print(f'pseudo-drop <= {t}: balanced mixed acc {sum(accs)/len(accs):.3f}')
print('--- only rows with variant samples (no gold-based shortcut)')
rows2=[r for r in rows if r[3] is not None]
per2=collections.defaultdict(list)
for r in rows2: per2[r[1]].append(r)
mixed2={p:v for p,v in per2.items() if len({x[2] for x in v})==2}
print('rows',len(rows2),'robust',sum(r[2] for r in rows2),'mixed problems',len(mixed2))
for t in (0.0,0.1,0.2,0.3):
    accs=[]
    for _ in range(2000):
        ok=n=0
        for p,v in mixed2.items():
            pos=[x for x in v if x[2]]; neg=[x for x in v if not x[2]]
            for r in (rng.choice(pos), rng.choice(neg)):
                ok+=(r[3][0]<=t)==r[2]; n+=1
        accs.append(ok/n)
    allacc=sum((r[3][0]<=t)==r[2] for r in rows2)/len(rows2)
    print(f'pseudo-drop <= {t}: balanced mixed {sum(accs)/len(accs):.3f}   all rows {allacc:.3f}')
