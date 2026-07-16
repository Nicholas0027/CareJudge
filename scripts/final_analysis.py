#!/usr/bin/env python3
"""Final analysis with significance testing + capability ladder (14-feature CARE).

Produces everything the revised paper needs:
  (1) Per-(judge,bench) AUROC mean +/- std over N seeds for CARE / SCOPE / ToE /
      majority-vote / weighted-vote and the three feature-group ablations.
  (2) Paired significance tests: for each (judge,bench), paired bootstrap 95% CI
      of AUROC(CARE) - AUROC(best single signal), plus a Wilcoxon signed-rank test
      across the 12 pairs for "CARE > best baseline".
  (3) Selective metrics (coverage/acc/violation) at alpha=0.15.
  (4) Capability-ladder summary: mean fused AUROC vs raw accuracy per judge tag.

Uses the CONTRIBUTED 14-feature vector (no sim_* dead features, no adaptive_calls).

Usage: python scripts/final_analysis.py --features-dir outputs/scale --out outputs/final_analysis.json --seeds 20
"""
import os, sys, json, random, argparse
import numpy as np
from scipy.stats import beta as beta_dist, wilcoxon
from sklearn.linear_model import LogisticRegression

# All judge tags we may have (ladder + API). Missing files are skipped.
JUDGES = [
    ("qwen-1.5b", "Qwen2.5-1.5B", "ladder"),
    ("qwen-7b",   "Qwen2.5-7B",   "ladder"),
    ("qwen-14b",  "Qwen2.5-14B",  "ladder"),
    ("deepseek-chat", "DeepSeek-V4", "api"),
    ("gpt-5_5",   "GPT-5.5",      "api"),
]
BENCHMARKS = ["judgebench", "tldr", "rewardbench", "lmaarena"]

# Contributed 14-feature vector (protocol + confidence + length; NO sim_*).
FEAT_PROTOCOL = ["feat_swap_consistency","feat_swap_conf_gap",
                 "feat_rubric_vote_share","feat_rubric_entropy","feat_rubric_flip",
                 "feat_self_vote_share","feat_self_entropy"]
FEAT_CONF = ["confidence","feat_base_conf","feat_mean_conf","feat_std_conf"]
FEAT_LENGTH = ["feat_length_gap_norm","feat_score_margin"]
FEAT_FULL = FEAT_PROTOCOL + FEAT_CONF + FEAT_LENGTH  # 14

def read_jsonl(path):
    rows=[]
    for l in open(path):
        l=l.strip()
        if not l: continue
        try:
            r=json.loads(l)
            if r.get("correct") is not None: rows.append(r)
        except: pass
    return rows

def to_matrix(rows, cols):
    return np.array([[float(r.get(c,0.0) or 0.0) for c in cols] for r in rows], dtype=np.float64)

def split3(n, seed, tr=0.4, ca=0.3):
    idx=list(range(n)); random.Random(seed).shuffle(idx)
    nt,nc=int(n*tr),int(n*ca)
    return idx[:nt], idx[nt:nt+nc], idx[nt+nc:]

def auroc(y, s):
    y=np.asarray(y); s=np.asarray(s)
    p=s[y==1]; n=s[y==0]
    if len(p)==0 or len(n)==0: return 0.5
    return float((np.sum(p[:,None]>n[None,:])+0.5*np.sum(p[:,None]==n[None,:]))/(len(p)*len(n)))

def fit_predict(Xtr, ytr, Xt):
    if len(set(ytr))<2: return np.full(len(Xt), float(np.mean(ytr)))
    clf=LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'); clf.fit(Xtr,ytr)
    return clf.predict_proba(Xt)[:,1]

def cal_thr(p_cal, y_cal, alpha, delta=0.10, min_keep=20):
    order=np.argsort(p_cal)[::-1]; ps=p_cal[order]; ys=y_cal[order]; n=len(ps)
    if n<min_keep: return float('inf')
    ce=np.cumsum(ys==0); pref=np.arange(min_keep,n+1); ev=ce[pref-1]
    mask=ev<pref; cp=np.ones(len(pref))
    if np.any(mask): cp[mask]=beta_dist.ppf(1-delta, ev[mask]+1, pref[mask]-ev[mask])
    valid=cp<=alpha
    if not np.any(valid): return float('inf')
    return float(ps[pref[valid][-1]-1])

def selective(p_cal, y_cal, p_test, y_test, alpha=0.15):
    thr=cal_thr(p_cal, y_cal, alpha)
    am=p_test>=thr; cov=float(np.mean(am))
    if np.any(am):
        acc=float(np.mean(y_test[am])); risk=1-acc
    else:
        acc=0.0; risk=0.0
    return cov, acc, (1 if risk>alpha else 0)

def scope_scores(rows):
    return np.array([(1.0-abs(float(r.get("feat_swap_conf_gap",0))))*float(r.get("feat_swap_consistency",0.5)) for r in rows])
def toe_scores(rows):
    return np.array([max(float(r.get("feat_rubric_vote_share",0.5)),float(r.get("feat_self_vote_share",0.5)),float(r.get("feat_swap_consistency",0.5))) for r in rows])
def base_scores(rows):
    return np.array([float(r.get("feat_base_conf",0.5)) for r in rows])
def rubric_scores(rows):
    return np.array([float(r.get("feat_rubric_vote_share",0.5)) for r in rows])
def self_scores(rows):
    return np.array([float(r.get("feat_self_vote_share",0.5)) for r in rows])
def maj_scores(rows):
    out=[]
    for r in rows:
        pc=r.get("per_call",{}); w=[pc.get("base_winner","")]
        for c in pc.get("self_calls",[]):
            if c.get("winner") in ("A","B"): w.append(c["winner"])
        sw=pc.get("swap_winner","")
        if sw in ("A","B"): w.append("B" if sw=="A" else "A")
        for c in pc.get("rubric_calls",[]):
            if c.get("winner") in ("A","B"): w.append(c["winner"])
        w=[x for x in w if x in ("A","B")]
        out.append(max(w.count("A"),w.count("B"))/len(w) if w else 0.5)
    return np.array(out)
def wtd_scores(rows):
    out=[]
    for r in rows:
        pc=r.get("per_call",{})
        calls=[("base",pc.get("base_winner",""),pc.get("base_conf",0.5))]
        for c in pc.get("self_calls",[]): calls.append(("self",c.get("winner",""),c.get("confidence",0.5)))
        sw=pc.get("swap_winner","")
        if sw in ("A","B"): calls.append(("swap","B" if sw=="A" else "A",pc.get("swap_conf",0.5)))
        for c in pc.get("rubric_calls",[]): calls.append(("rubric",c.get("winner",""),c.get("confidence",0.5)))
        wl=[w for _,w,_ in calls if w in ("A","B")]
        if not wl: out.append(0.5); continue
        modal="A" if wl.count("A")>=wl.count("B") else "B"
        cs=[c for _,w,c in calls if w==modal]
        out.append(float(np.mean(cs)) if cs else 0.5)
    return np.array(out)

def analyze(path, seeds=20, alpha=0.15):
    rows=read_jsonl(path)
    if len(rows)<50: return None
    raw=float(np.mean([r["correct"] for r in rows])); n=len(rows)
    X={"full":to_matrix(rows,FEAT_FULL),"proto":to_matrix(rows,FEAT_PROTOCOL),
       "conf":to_matrix(rows,FEAT_CONF),"len":to_matrix(rows,FEAT_LENGTH)}
    y=np.array([int(r["correct"]) for r in rows])
    nonfit={"scope":scope_scores(rows),"toe":toe_scores(rows),"base":base_scores(rows),
            "rubric":rubric_scores(rows),"self":self_scores(rows),
            "maj_vote":maj_scores(rows),"wtd_vote":wtd_scores(rows)}
    methods=["care","scope","toe","base","rubric","self","maj_vote","wtd_vote",
             "abl_proto","abl_conf","abl_len"]
    agg={m:{"auroc":[],"cov":[],"acc":[],"viol":[]} for m in methods}
    # store per-seed CARE and best-single for paired test
    care_seed=[]; bestsingle_seed=[]
    for seed in range(seeds):
        tr,ca,te=split3(n,seed)
        if len(te)<10: continue
        yc=y[ca]; yt=y[te]
        pc=fit_predict(X["full"][tr],y[tr],X["full"][ca]); pt=fit_predict(X["full"][tr],y[tr],X["full"][te])
        a=auroc(yt,pt); agg["care"]["auroc"].append(a)
        cov,acc,v=selective(pc,yc,pt,yt,alpha); agg["care"]["cov"].append(cov); agg["care"]["acc"].append(acc); agg["care"]["viol"].append(v)
        care_seed.append(a)
        # single signals (non-fit)
        singles={}
        for nm in ["scope","toe","base","rubric","self","maj_vote","wtd_vote"]:
            s=nonfit[nm]; pcb=s[ca]; ptb=s[te]
            ab=auroc(yt,ptb); agg[nm]["auroc"].append(ab)
            cov,acc,v=selective(pcb,yc,ptb,yt,alpha); agg[nm]["cov"].append(cov); agg[nm]["acc"].append(acc); agg[nm]["viol"].append(v)
            singles[nm]=ab
        # best single among the pure confidence signals (base/rubric/self/scope)
        bestsingle_seed.append(max(singles["base"],singles["rubric"],singles["self"],singles["scope"]))
        # ablations
        for nm,key in [("abl_proto","proto"),("abl_conf","conf"),("abl_len","len")]:
            pca=fit_predict(X[key][tr],y[tr],X[key][ca]); pta=fit_predict(X[key][tr],y[tr],X[key][te])
            agg[nm]["auroc"].append(auroc(yt,pta))
            cov,acc,v=selective(pca,yc,pta,yt,alpha); agg[nm]["cov"].append(cov); agg[nm]["acc"].append(acc); agg[nm]["viol"].append(v)
    def m(x): return float(np.mean(x)) if x else None
    def sd(x): return float(np.std(x)) if x else 0.0
    out={"n_total":n,"raw_acc":raw,"seeds":seeds}
    for meth in methods:
        out[meth]={"auroc_mean":m(agg[meth]["auroc"]),"auroc_std":sd(agg[meth]["auroc"]),
                   "cov":m(agg[meth]["cov"]),"acc":m(agg[meth]["acc"]),"viol":m(agg[meth]["viol"])}
    # paired bootstrap CI of CARE - best_single (per-seed deltas)
    if care_seed and bestsingle_seed:
        deltas=np.array(care_seed)-np.array(bestsingle_seed)
        boot=[np.mean(np.random.choice(deltas,len(deltas),replace=True)) for _ in range(2000)]
        out["care_minus_bestsingle"]={"mean":float(np.mean(deltas)),
            "ci95":[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))]}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features-dir",default="outputs/scale")
    ap.add_argument("--out",default="outputs/final_analysis.json")
    ap.add_argument("--seeds",type=int,default=20)
    args=ap.parse_args()
    np.random.seed(0)
    results={}
    care_wins=[]  # per-pair CARE auroc
    best_base=[]  # per-pair best baseline auroc (for Wilcoxon)
    for tag,name,kind in JUDGES:
        results[tag]={"name":name,"kind":kind,"benches":{}}
        for b in BENCHMARKS:
            p=os.path.join(args.features_dir,f"{tag}_{b}_features.jsonl")
            if not os.path.exists(p): continue
            r=analyze(p,seeds=args.seeds)
            if r: 
                results[tag]["benches"][b]=r
                # collect for global Wilcoxon: CARE vs best of {base,scope,maj_vote,wtd_vote}
                bb=max(r["base"]["auroc_mean"],r["scope"]["auroc_mean"],r["maj_vote"]["auroc_mean"],r["wtd_vote"]["auroc_mean"])
                care_wins.append(r["care"]["auroc_mean"]); best_base.append(bb)
                print(f"{name}/{b}: raw={r['raw_acc']:.3f} CARE={r['care']['auroc_mean']:.3f}+-{r['care']['auroc_std']:.3f} "
                      f"scope={r['scope']['auroc_mean']:.3f} maj={r['maj_vote']['auroc_mean']:.3f} wtd={r['wtd_vote']['auroc_mean']:.3f} "
                      f"delta_ci={r.get('care_minus_bestsingle',{}).get('ci95')}", flush=True)
    # Global Wilcoxon
    if len(care_wins)>=6:
        stat,pval=wilcoxon(care_wins,best_base,alternative='greater')
        results["_global_wilcoxon"]={"n_pairs":len(care_wins),"statistic":float(stat),"p_value":float(pval),
            "n_care_wins":int(np.sum(np.array(care_wins)>np.array(best_base)))}
        print(f"\nGlobal Wilcoxon (CARE > best baseline over {len(care_wins)} pairs): W={stat:.1f} p={pval:.4g} "
              f"wins={results['_global_wilcoxon']['n_care_wins']}/{len(care_wins)}", flush=True)
    json.dump(results, open(args.out,"w"), indent=2)
    print(f"\nSaved -> {args.out}")
    # Capability ladder table
    print("\n=== CAPABILITY LADDER (same-family Qwen2.5, controlled size) ===")
    print(f"{'Model':<16} {'params':>7} {'mean raw':>9} {'mean CARE AUROC':>16}")
    for tag,name,kind in JUDGES:
        if kind!="ladder": continue
        bs=results[tag]["benches"]
        if not bs: continue
        raws=[v["raw_acc"] for v in bs.values()]; cares=[v["care"]["auroc_mean"] for v in bs.values()]
        print(f"{name:<16} {'':<7} {np.mean(raws):>9.3f} {np.mean(cares):>16.3f}")

if __name__=="__main__":
    main()
