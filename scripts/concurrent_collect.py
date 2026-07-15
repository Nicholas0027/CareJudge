#!/usr/bin/env python3
import os,sys,json,time;sys.path.insert(0,'/data/lab/CareJudge')
os.chdir('/data/lab/CareJudge')
for l in open('.env'):
    if '=' in l and not l.startswith('#'):k,v=l.strip().split('=',1);os.environ[k]=v
from concurrent.futures import ThreadPoolExecutor, as_completed
from care_judge.schemas import PairItem
from care_judge.uncertainty.collector_extended import collect_with_call_trace
from openai import OpenAI

JUDGE_NAME = sys.argv[1]
BENCH_NAME = sys.argv[2]
R = ["You are an expert evaluator. Compare Response A and Response B. Output ONLY the letter A or B.",
     "Which response is better? Compare A and B carefully and output ONLY A or B.",
     "Compare the two responses. Which one would a careful evaluator prefer? Output ONLY A or B."]
items = [PairItem(id=r['id'],prompt=r['prompt'],response_a=r['response_a'],response_b=r['response_b'],label=r.get('label'))
         for r in [json.loads(l) for l in open(f'data/{BENCH_NAME}_2k.jsonl') if l.strip()]]
pfx = {'ds':'deepseek-chat','gpt55':'gpt-5_5','glm':'glm-5.2','qwen':'qwen-1.5b'}[JUDGE_NAME]
if JUDGE_NAME=='ds':
    from care_judge.judges.deepseek_compat import DeepSeekJudge
    j=DeepSeekJudge(model='deepseek-chat')
elif JUDGE_NAME=='gpt55':
    from care_judge.judges.openai_compat import OpenAICompatJudge
    j=OpenAICompatJudge(model='gpt-5.5')
elif JUDGE_NAME=='glm':
    class G:
        def judge(self, item, rubric, temp):
            from care_judge.schemas import Judgment
            resp = self.client.chat.completions.create(model=self.model, messages=[{"role":"user","content":f"{rubric}\n\nA: {item.response_a}\nB: {item.response_b}"}], temperature=temp, max_tokens=5)
            ans = resp.choices[0].message.content.strip().upper()
            w = "A" if "A" in ans else "B"
            return Judgment(winner=w, confidence=0.99, cost=0.0)
        def __init__(s):s.client=OpenAI(api_key=os.environ['GLM_API_KEY'],base_url="https://open.bigmodel.cn/api/paas/v4");s.model='glm-5.2'
    j=G()
elif JUDGE_NAME=='qwen':
    from care_judge.judges.local_hf import LocalHFJudge
    j=LocalHFJudge(model_name='Qwen/Qwen2.5-1.5B-Instruct')

out=f'outputs/scale/{pfx}_{BENCH_NAME}_features.jsonl'
os.makedirs('outputs/scale',exist_ok=True)
done=sum(1 for _ in open(out)) if os.path.exists(out) else 0; items=items[done:]
def process(it):
    try:return it.id,collect_with_call_trace(it,j,R,k_self=3,temperature=0.7)
    except:return it.id,None
t0=time.time()
with ThreadPoolExecutor(max_workers=5)as ex:
    futs=[ex.submit(process,it)for it in items]
    with open(out,'a')as f:
        for i,fut in enumerate(as_completed(futs)):
            try:
                iid,row=fut.result()
                if row:f.write(json.dumps(row)+'\n');f.flush()
            except:pass
            if(i+1)%50==0:print(f'  {done+i+1}/{done+len(items)}',flush=True)
elapsed=time.time()-t0
print(f'DONE {JUDGE_NAME}/{BENCH_NAME} {len(items)} in {elapsed/60:.1f}min ({len(items)/elapsed*3600:.0f}/h)',flush=True)
