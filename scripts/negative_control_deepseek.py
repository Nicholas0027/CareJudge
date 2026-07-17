#!/usr/bin/env python3
"""DeepSeek-V4 negative-control: equivalent vs non-equivalent rubrics on JudgeBench.

Uses the DeepSeek API (deepseek-chat) to run the SAME negative-control design
as the 1.5B experiment, but on a mid-capability judge where the signal is
expected to be meaningful. If non-equivalent rubrics produce >> higher flip
rates than equivalent ones on DeepSeek, the rubric-stability signal is
validated as reflecting genuine judge instability rather than criterion changes.

Usage: python scripts/negative_control_deepseek.py
"""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Equivalent rubrics (wording varies, criterion = "which is better" held fixed)
EQUIV_RUBRICS = [
    "You are an expert evaluator. Compare Response A and Response B. Output ONLY the letter A or B.",
    "Which response is better? Compare A and B carefully and output ONLY A or B.",
    "Compare the two responses. Which one would a careful evaluator prefer? Output ONLY A or B.",
]

# Non-equivalent rubrics (criterion itself changes: brevity vs thoroughness vs formality)
NONEQUIV_RUBRICS = [
    "Which response is more CONCISE? Prefer shorter, more direct answers. Compare A and B. Output ONLY A or B.",
    "Which response is more THOROUGH? Prefer longer, more detailed answers. Compare A and B. Output ONLY A or B.",
    "Which response is more FORMAL? Prefer professional tone and structure. Compare A and B. Output ONLY A or B.",
]

JUDGE_PROMPT_TEMPLATE = """You are an impartial evaluator. Judge which response is better.

Rubric:
{rubric}

User prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Return only valid JSON with keys:
{{"winner": "A" or "B", "confidence": a number from 0 to 1, "reason": "brief explanation"}}
"""

def parse_winner(text):
    text = text.strip()
    try:
        obj = json.loads(text)
        w = str(obj.get("winner","")).strip().upper()
        if w in ("A","B"): return w
    except: pass
    import re
    m = re.search(r'\{.*\}', text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            w = str(obj.get("winner","")).strip().upper()
            if w in ("A","B"): return w
        except: pass
    if re.search(r'\bA\b', text, flags=re.I) and not re.search(r'\bB\b', text, flags=re.I):
        return "A"
    elif re.search(r'\bB\b', text, flags=re.I) and not re.search(r'\bA\b', text, flags=re.I):
        return "B"
    return "invalid"

def main():
    # Load env
    for l in open('.env'):
        if '=' in l and not l.startswith('#'):
            k,v = l.strip().split('=',1); os.environ[k]=v

    client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url='https://api.deepseek.com/v1')

    # Load JudgeBench items
    items = [json.loads(l) for l in open('data/judgebench_2k.jsonl') if l.strip()]
    print(f"JudgeBench: {len(items)} items", flush=True)

    def call_judge(item, rubric):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            rubric=rubric, prompt=item['prompt'],
            response_a=item['response_a'], response_b=item['response_b']
        )
        try:
            r = client.chat.completions.create(
                model='deepseek-chat',
                messages=[{"role":"user","content":prompt}],
                temperature=0, max_tokens=80
            )
            return parse_winner(r.choices[0].message.content)
        except Exception as e:
            return "error"

    def run_condition(rubrics, label):
        flips = 0; total = 0
        # Use thread pool for speed (API calls are I/O bound)
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {}
            for i, item in enumerate(items):
                for j, rubric in enumerate(rubrics):
                    futs[ex.submit(call_judge, item, rubric)] = (i, j)

            # Collect results: verdicts[item_idx] = [v0, v1, v2]
            verdicts = {}
            for fut in as_completed(futs):
                item_idx, rubric_idx = futs[fut]
                v = fut.result()
                if item_idx not in verdicts:
                    verdicts[item_idx] = [None]*len(rubrics)
                verdicts[item_idx][rubric_idx] = v

            for idx, vs in verdicts.items():
                valid = [v for v in vs if v in ("A","B")]
                if len(valid) >= 2:
                    total += 1
                    if len(set(valid)) > 1:
                        flips += 1
            if (flips + total) % 100 == 0 or flips + total > 0:
                pass  # progress
        rate = flips / total if total > 0 else 0
        print(f"  {label}: {flips}/{total} flipped ({rate:.1%})", flush=True)
        return {'label': label, 'flips': flips, 'total': total, 'flip_rate': rate}

    print("\n=== Equivalent rubrics (wording perturbation) ===", flush=True)
    equiv = run_condition(EQUIV_RUBRICS, "equivalent")

    print("\n=== Non-equivalent rubrics (criterion perturbation) ===", flush=True)
    nonequiv = run_condition(NONEQUIV_RUBRICS, "non-equivalent")

    result = {
        'judge': 'DeepSeek-V4 (deepseek-chat)',
        'benchmark': 'JudgeBench (620 items)',
        'equivalent': equiv,
        'non_equivalent': nonequiv,
        'ratio': nonequiv['flip_rate'] / equiv['flip_rate'] if equiv['flip_rate'] > 0 else float('inf'),
    }
    if nonequiv['flip_rate'] > equiv['flip_rate'] * 1.5:
        result['interpretation'] = 'Non-equivalent rubrics produce substantially higher flip rates, confirming that equivalent-rubric flips reflect judge instability rather than criterion changes.'
    else:
        result['interpretation'] = 'Flip rates are similar — see discussion.'

    print(f"\n=== RESULT ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)

    os.makedirs('outputs/supplementary', exist_ok=True)
    json.dump(result, open('outputs/supplementary/negative_control_deepseek.json', 'w'), indent=2)
    print(f"\nSaved -> outputs/supplementary/negative_control_deepseek.json", flush=True)

if __name__ == '__main__':
    main()
