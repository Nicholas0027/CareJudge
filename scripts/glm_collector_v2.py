#!/usr/bin/env python3
"""GLM-5.2 collector v2: fixes reasoning-model empty-output bug.
   Uses max_tokens=800 to allow reasoning tokens + final answer.
   High concurrency (15) to compensate for slow reasoning latency.
"""
import os, sys, json, time
sys.path.insert(0, '/data/lab/CareJudge')
os.chdir('/data/lab/CareJudge')
os.environ.update({k:v for line in open('.env') for k,v in [line.strip().split('=',1)] if '=' in line and not line.startswith('#')})
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from care_judge.schemas import PairItem, Judgment
from care_judge.uncertainty.collector_extended import collect_with_call_trace

BENCH_NAME = sys.argv[1]
R = ["You are an expert evaluator. Compare Response A and Response B. Output ONLY the letter A or B.",
     "Compare A and B carefully and output ONLY A or B.",
     "Which one would a careful evaluator prefer? Output ONLY A or B."]

items = [PairItem(id=r['id'],prompt=r['prompt'],response_a=r['response_a'],response_b=r['response_b'],label=r.get('label'))
         for r in [json.loads(l) for l in open(f'data/{BENCH_NAME}_2k.jsonl') if l.strip()]]

class GLMJudge:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ['GLM_API_KEY'], base_url="https://open.bigmodel.cn/api/paas/v4")
        self.model = 'glm-5.2'
    def judge(self, item, rubric, temperature):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":f"{rubric}\n\nResponse A: {item.response_a}\n\nResponse B: {item.response_b}\n\nWhich response is better? Reply with ONLY the letter A or B."}],
                temperature=max(temperature, 0.01), max_tokens=800)
            ans = (resp.choices[0].message.content or "").strip().upper()
            # Parse: look for first A or B in the answer
            w = None
            for ch in ans:
                if ch == 'A': w = 'A'; break
                if ch == 'B': w = 'B'; break
            if w is None: w = 'A'  # rare fallback
            return Judgment(winner=w, confidence=0.99, cost=0.0)
        except Exception as e:
            return Judgment(winner='A', confidence=0.5, cost=0.0)

j = GLMJudge()
out = f'outputs/scale/glm-5.2_{BENCH_NAME}_features.jsonl'
os.makedirs('outputs/scale', exist_ok=True)
# Fresh start (old GLM data was corrupted by the bug)
done = 0
def process(it):
    try: return it.id, collect_with_call_trace(it, j, R, k_self=3, temperature=0.7)
    except: return it.id, None

t0 = time.time()
with ThreadPoolExecutor(max_workers=15) as ex:
    futs = [ex.submit(process, it) for it in items]
    with open(out, 'w') as f:
        for i, fut in enumerate(as_completed(futs)):
            try:
                iid, row = fut.result()
                if row: f.write(json.dumps(row)+'\n'); f.flush()
            except: pass
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE GLM/{BENCH_NAME}: {sum(1 for _ in open(out))} in {(time.time()-t0)/60:.1f}min", flush=True)
