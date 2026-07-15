
import os,sys,json,time;sys.path.insert(0,'/data/lab/CareJudge')
os.chdir('/data/lab/CareJudge')
os.environ.update({k:v for k,v in [l.strip().split('=',1) for l in open('.env') if '=' in l and not l.startswith('#')]})
from concurrent.futures import ThreadPoolExecutor, as_completed
from care_judge.schemas import PairItem, Judgment
from openai import OpenAI

JUDGE_NAME='glm'; BENCH_NAME=sys.argv[1]
R=["You are an expert evaluator. Compare Response A and Response B. Output ONLY the letter A or B.",
   "Compare A and B carefully and output ONLY A or B.",
   "Which one would a careful evaluator prefer? Output ONLY A or B."]

items = [PairItem(id=r['id'],prompt=r['prompt'],response_a=r['response_a'],response_b=r['response_b'],label=r.get('label'))
         for r in [json.loads(l) for l in open(f'data/{BENCH_NAME}_2k.jsonl') if l.strip()]]

class GLMJudge:
    def __init__(self):self.client=OpenAI(api_key=os.environ['GLM_API_KEY'],base_url="https://open.bigmodel.cn/api/paas/v4");self.model='glm-5.2'
    def judge(self, item, rubric, temperature):
        resp=self.client.chat.completions.create(model=self.model,messages=[{"role":"user","content":f"{rubric}\n\nA: {item.response_a}\nB: {item.response_b}"}],temperature=temperature,max_tokens=5)
        ans=resp.choices[0].message.content.strip().upper()
        return Judgment(winner="A" if "A" in ans else "B",confidence=0.99,cost=0.0)

from care_judge.uncertainty.collector_extended import collect_with_call_trace
j=GLMJudge()
out=f'outputs/scale/glm-5.2_{BENCH_NAME}_features.jsonl'
os.makedirs('outputs/scale',exist_ok=True)
done=sum(1 for _ in open(out)) if os.path.exists(out) else 0;items=items[done:]
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
print(f'DONE GLM/{BENCH_NAME} {len(items)} in {elapsed/60:.1f}min',flush=True)
