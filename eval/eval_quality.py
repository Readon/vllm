#!/usr/bin/env python3
"""
Comprehensive model quality evaluation across different vLLM configurations.

Evaluates 6 dimensions:
  1. Logprob fidelity     — token-level logprob correlation vs baseline
  2. Greedy consistency   — exact match of greedy decoding output
  3. QA accuracy          — factual short-answer correctness (CN + EN)
  4. Tool call            — structured function calling format + argument accuracy
  5. Bug finding          — code bug identification accuracy
  6. Reasoning            — multi-step math/logic chain-of-thought quality

Usage:
  # Start server with config A (baseline)
  python eval_quality.py --url http://localhost:8000 --output eval/baseline.json

  # Restart with config B
  python eval_quality.py --url http://localhost:8000 --output eval/config_B.json --baseline eval/baseline.json

  # Skip slow tests
  python eval_quality.py --url http://localhost:8000 --output eval/quick.json --skip-qa --skip-reasoning
"""

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "/model_cache/snapshots/56f41874389615226dcd849ded92261a0286ff59"

# ═══════════════════════════════════════════════════════════════════════════
#  Test Data
# ═══════════════════════════════════════════════════════════════════════════

LOGPROB_PROMPTS = [
    # Chinese factual
    "中国的首都是",
    "地球上最大的海洋是",
    "光在真空中的速度约为",
    "水的化学式是",
    "《红楼梦》的作者是",
    "太阳系中最大的行星是",
    "人体最大的器官是",
    "DNA的中文全称是",
    # English factual
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at",
    "The speed of light is approximately",
    "The author of Romeo and Juliet is",
    "The chemical formula for table salt is",
    "The tallest mountain on Earth is",
    "The chemical symbol for gold is",
    # Math / Reasoning
    "If a train travels at 60 km/h for 2.5 hours, it covers",
    "The square root of 144 is",
    "In binary, the decimal number 10 is written as",
    "2 + 2 equals",
    # Continuation
    "Please explain briefly why the sky appears blue during the day:",
    "In one word, the process by which plants convert sunlight to energy is called",
    "The Boolean value of an empty list in Python is",
    "The result of 15 modulo 4 is",
]

QA_DATASET = [
    # Chinese
    {"q": "中国的首都是哪个城市？", "a": ["北京", "Beijing"]},
    {"q": "地球围绕太阳公转一圈需要多长时间？", "a": ["一年", "365", "365天", "one year", "1年", "约365天"]},
    {"q": "水的沸点是多少摄氏度（标准大气压下）？", "a": ["100"]},
    {"q": "世界上最高的山峰叫什么？", "a": ["珠穆朗玛峰", "珠峰", "Everest"]},
    {"q": "光合作用需要什么气体？", "a": ["二氧化碳", "CO2", "co2"]},
    {"q": "地球自转一圈需要多长时间？", "a": ["24小时", "24", "一天", "24小", "二十四小时"]},
    {"q": "氧气的化学式是什么？", "a": ["O2", "o2", "O₂", "O₂", "o₂"]},
    {"q": "一年有多少个月？", "a": ["12", "十二"]},
    {"q": "月球绕地球一圈大约需要多少天？", "a": ["27", "28", "27.3"]},
    {"q": "一千克等于多少克？", "a": ["1000", "一千"]},
    # English
    {"q": "What is the capital of Japan?", "a": ["Tokyo", "tokyo", "东京", "Tōkyō"]},
    {"q": "What is the largest ocean on Earth?", "a": ["Pacific", "pacific", "太平洋", "pacific ocean"]},
    {"q": "How many continents are there?", "a": ["seven", "Seven", "7"]},
    {"q": "What planet is known as the Red Planet?", "a": ["Mars", "mars", "火星"]},
    {"q": "What is the chemical symbol for gold?", "a": ["Au", "au", "金"]},
    {"q": "Who wrote the theory of relativity?", "a": ["Einstein", "einstein", "爱因斯坦", "Albert Einstein"]},
    {"q": "What is the boiling point of water in Celsius?", "a": ["100"]},
    {"q": "How many sides does a hexagon have?", "a": ["six", "Six", "6"]},
    {"q": "What is the largest mammal?", "a": ["blue whale", "Blue whale", "蓝鲸"]},
    {"q": "What gas do humans exhale?", "a": ["carbon dioxide", "CO2", "二氧化碳", "co2"]},
]

TOOL_CALL_TESTS = [
    {
        "name": "get_weather",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. Beijing"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        }],
        "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
        "expected_function": "get_weather",
        "expected_args": {"location": "北京", "unit": "celsius"},
        "required_args": ["location"],
        "arg_checks": {"location": "北京"},
    },
    {
        "name": "search",
        "tools": [{
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "num_results": {"type": "integer", "description": "Number of results"}
                    },
                    "required": ["query"]
                }
            }
        }],
        "messages": [{"role": "user", "content": "帮我搜索一下量子计算的最新进展"}],
        "expected_function": "search_web",
        "expected_args": {"query": "量子计算"},
        "required_args": ["query"],
        "arg_checks": {"query": "量子"},
    },
    {
        "name": "calculate",
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression to evaluate"},
                    },
                    "required": ["expression"]
                }
            }
        }],
        "messages": [{"role": "user", "content": "What is 123 * 456?"}],
        "expected_function": "calculate",
        "expected_args": {"expression": "123 * 456"},
        "required_args": ["expression"],
        "arg_checks": {"expression": "123"},
    },
    {
        "name": "multi_tool_choice",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send an email to a recipient",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string", "description": "Recipient email"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "subject", "body"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "messages": [{"role": "user", "content": "请给 alice@example.com 发一封邮件，主题是项目更新，内容是项目进展顺利。"}],
        "expected_function": "send_email",
        "expected_args": {"to": "alice@example.com", "subject": "项目更新"},
        "required_args": ["to", "subject", "body"],
        "arg_checks": {"to": "alice", "subject": "项目更新"},
    },
]

BUG_FINDER_TESTS = [
    {
        "name": "off_by_one",
        "code": '''def get_first_n(items, n):
    result = []
    for i in range(1, n + 1):
        result.append(items[i])
    return result''',
        "bug_description": "Index starts at 1 instead of 0, skipping first element and potentially causing IndexError",
        "severity": "high",
    },
    {
        "name": "infinite_loop",
        "code": '''def count_down(n):
    while n > 0:
        print(n)
        # missing: n -= 1
    return "Done!"''',
        "bug_description": "Infinite loop because n is never decremented",
        "severity": "high",
    },
    {
        "name": "mutable_default",
        "code": '''def add_item(item, items=[]):
    items.append(item)
    return items''',
        "bug_description": "Mutable default argument shared across calls",
        "severity": "medium",
    },
    {
        "name": "wrong_comparison",
        "code": '''def is_equal(a, b):
    if a = b:
        return True
    return False''',
        "bug_description": "Assignment (=) instead of comparison (==)",
        "severity": "high",
    },
    {
        "name": "division_by_zero",
        "code": '''def average(numbers):
    total = sum(numbers)
    return total / len(numbers)''',
        "bug_description": "Division by zero when the list is empty",
        "severity": "medium",
    },
    {
        "name": "resource_leak",
        "code": '''def read_file(path):
    f = open(path, 'r')
    data = f.read()
    return data''',
        "bug_description": "File handle not closed, resource leak",
        "severity": "medium",
    },
    {
        "name": "incorrect_sort",
        "code": '''def sort_dict_by_value(d):
    return sorted(d.items(), key=lambda x: x[1], reverse=True)

# Expected: ascending order, but reverse=True gives descending''',
        "bug_description": "reverse=True gives descending order when ascending was expected",
        "severity": "low",
    },
    {
        "name": "scope_issue",
        "code": '''total = 0
def add_to_total(x):
    total += x
    return total''',
        "bug_description": "UnboundLocalError: local variable 'total' referenced before assignment",
        "severity": "high",
    },
    {
        "name": "type_confusion",
        "code": '''def concat(a, b):
    return a + b

# Usage: concat("Age: ", 25) will raise TypeError''',
        "bug_description": "No type checking; will fail with TypeError when mixing str and int",
        "severity": "medium",
    },
    {
        "name": "correct_code",
        "code": '''def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b''',
        "bug_description": None,  # This is correct code, model should say no bug
        "severity": None,
    },
]

REASONING_TESTS = [
    {
        "name": "multi_step_arithmetic",
        "question": "A store sells apples for $2 each and oranges for $3 each. If Alice buys 4 apples and 3 oranges, and pays with a $50 bill, how much change does she get?",
        "answer": "50 - (4*2 + 3*3) = 50 - 17 = 33",
        "check": lambda r: "33" in r,
    },
    {
        "name": "logic_puzzle",
        "question": "All cats have tails. Fluffy is a cat. Does Fluffy have a tail? Answer yes or no and explain why.",
        "answer": "Yes, by syllogism",
        "check": lambda r: "yes" in r.lower() and ("cat" in r.lower() or "syllog" in r.lower() or "all" in r.lower()),
    },
    {
        "name": "rate_problem",
        "question": "If 5 workers can paint 5 houses in 5 days, how many days would it take 10 workers to paint 10 houses?",
        "answer": "5 days",
        "check": lambda r: "5" in r and ("day" in r.lower()),
    },
    {
        "name": "probability",
        "question": "A fair coin is flipped 3 times. What is the probability of getting exactly 2 heads? Give the answer as a fraction.",
        "answer": "3/8",
        "check": lambda r: "3/8" in r or "3 / 8" in r or "0.375" in r,
    },
    {
        "name": "spatial_reasoning",
        "question": "A cube has 6 faces. If you paint all faces red and then cut it into 8 smaller equal cubes, how many small cubes will have exactly 3 red faces?",
        "answer": "8",
        "check": lambda r: "8" in r.split(".")[0].split(",")[0] and ("all" in r.lower() or "every" in r.lower() or "each" in r.lower() or "corner" in r.lower()),
    },
    {
        "name": "sequence",
        "question": "What is the next number in the sequence: 2, 6, 12, 20, 30, ?",
        "answer": "42",
        "check": lambda r: "42" in r,
    },
    {
        "name": "age_problem",
        "question": "Tom is twice as old as Mary. In 5 years, Tom will be 25 years old. How old is Mary now?",
        "answer": "10",
        "check": lambda r: "10" in r,
    },
    {
        "name": "container_mixing",
        "question": "You have a 3-liter jug and a 5-liter jug. How can you measure exactly 4 liters of water? Describe the steps.",
        "answer": "Fill 5L, pour into 3L, leaving 2L in 5L. Empty 3L, pour 2L from 5L into 3L. Fill 5L again, pour 1L into 3L (which has 2L). 5L now has 4L.",
        "check": lambda r: "4" in r and ("5" in r) and ("3" in r) and ("pour" in r.lower() or "倒" in r or "fill" in r.lower() or "装" in r),
    },
]


# ═══════════════════════════════════════════════════════════════════════════
#  API Helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_max_model_len(url):
    """Query server for max_model_len."""
    try:
        req = urllib.request.Request(url + "/v1/models")
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        if data.get("data"):
            return data["data"][0].get("max_model_len", 4096)
    except Exception:
        pass
    return 4096


def api_completions(url, prompt, max_tokens=64, temperature=0, logprobs=5):
    data = json.dumps({
        "model": MODEL, "prompt": prompt,
        "temperature": temperature, "max_tokens": max_tokens, "logprobs": logprobs,
    }).encode()
    req = urllib.request.Request(url + "/v1/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


def api_chat(url, messages, max_tokens=256, temperature=0, tools=None):
    # Clamp max_tokens to leave room for prompt
    max_len = get_max_model_len(url)
    clamped = min(max_tokens, max(64, max_len - 256))
    body = {
        "model": MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": clamped,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


def extract_response(msg):
    """Extract the actual answer from a chat response message.

    Handles three cases:
    1. Qwen3 with reasoning parser: reasoning in msg['reasoning'], answer in msg['content']
    2. Qwen3 without reasoning parser: <thinkk>...</thinkk> blocks in content
    3. Plain response: content directly
    """
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""

    # If reasoning parser split them, content has the answer
    if reasoning and content:
        return content.strip()

    # If content has <thinkk> blocks, strip them
    text = content
    while "<thinkk>" in text and "</thinkk>" in text:
        pre = text[:text.find("<thinkk>")]
        post = text[text.find("</thinkk>") + len("</thinkk>"):]
        text = pre + post
    if "<thinkk>" in text:
        text = text[:text.find("<thinkk>")]

    # If content is empty after stripping (or was null), search the full reasoning
    # for the answer
    if not text.strip() and reasoning:
        # Return the full reasoning text so the checker can scan it
        return reasoning.strip()

    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluator Functions
# ═══════════════════════════════════════════════════════════════════════════

def eval_logprob(url, prompts):
    """Evaluate token logprobs and greedy outputs."""
    results = []
    for p in prompts:
        try:
            r = api_completions(url, p, max_tokens=32, temperature=0, logprobs=5)
            ch = r["choices"][0]
            token_lps = []
            top_toks = []
            if ch.get("logprobs") and ch["logprobs"].get("top_logprobs"):
                for tp in ch["logprobs"]["top_logprobs"]:
                    if tp is None:
                        continue
                    sorted_t = sorted(tp.items(), key=lambda x: -x[1])
                    top_toks.append(sorted_t)
                    if sorted_t:
                        token_lps.append(sorted_t[0][1])
            results.append({
                "prompt": p, "generated": ch["text"],
                "token_logprobs": token_lps, "top_tokens": top_toks,
                "num_tokens": r["usage"]["completion_tokens"],
            })
        except Exception as e:
            results.append({"prompt": p, "error": str(e)})
    return results


def eval_qa(url, dataset):
    """Evaluate factual QA accuracy."""
    results = []
    for item in dataset:
        try:
            r = api_chat(url, [{"role": "user", "content": item["q"] + " 请简短回答。"}],
                         max_tokens=1024, temperature=0)
            msg = r["choices"][0]["message"]
            text = extract_response(msg)
            correct = any(a.lower() in text.lower() for a in item["a"])
            results.append({
                "question": item["q"], "expected": item["a"],
                "response": text[:200], "correct": correct,
            })
        except Exception as e:
            results.append({"question": item["q"], "expected": item["a"],
                            "error": str(e), "correct": False})
    return results


def eval_tool_call(url, tests):
    """Evaluate tool calling accuracy."""
    results = []
    for t in tests:
        try:
            r = api_chat(url, t["messages"], max_tokens=512, temperature=0, tools=t["tools"])
            msg = r["choices"][0]["message"]

            # Check if tool_calls present
            tool_calls = msg.get("tool_calls", [])
            has_tool_call = len(tool_calls) > 0

            # Check function name
            fn_match = False
            args_ok = {}
            if has_tool_call:
                tc = tool_calls[0]
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_match = fn_name == t["expected_function"]

                # Check arguments
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                for key, expected_val in t["arg_checks"].items():
                    args_ok[key] = key in args and expected_val.lower() in str(args[key]).lower()

            # Parseable JSON check
            json_parseable = False
            if has_tool_call:
                try:
                    json.loads(tool_calls[0]["function"]["arguments"])
                    json_parseable = True
                except (json.JSONDecodeError, KeyError):
                    pass

            all_args_ok = all(args_ok.values()) if args_ok else False

            results.append({
                "name": t["name"],
                "has_tool_call": has_tool_call,
                "function_match": fn_match,
                "json_parseable": json_parseable,
                "args_ok": args_ok,
                "all_args_ok": all_args_ok,
                "required_args_present": all(
                    k in str(tool_calls) for k in t["required_args"]
                ) if has_tool_call else False,
                "raw_response": json.dumps({"content": msg.get("content"), "reasoning": (msg.get("reasoning") or "")[:100], "tool_calls": msg.get("tool_calls", [])}, ensure_ascii=False)[:300],
            })
        except Exception as e:
            results.append({
                "name": t["name"], "error": str(e),
                "has_tool_call": False, "function_match": False,
                "json_parseable": False, "all_args_ok": False,
            })
    return results


def eval_bug_finder(url, tests):
    """Evaluate bug identification accuracy."""
    results = []
    for t in tests:
        prompt = f"""Please analyze the following Python code and identify any bugs.

```python
{t['code']}
```

If you find a bug, describe it briefly. If there is no bug, say "No bug found".

Your analysis:"""
        try:
            r = api_chat(url, [{"role": "user", "content": prompt}],
                         max_tokens=512, temperature=0)
            text = extract_response(r["choices"][0]["message"])

            if t["bug_description"] is None:
                # Correct code — model should say no bug
                identified_correct = "no bug" in text.lower() or "没有" in text or "正确" in text or "代码是正确" in text
                false_positive = not identified_correct and ("bug" in text.lower() or "错误" in text or "问题" in text)
            else:
                # Buggy code — model should identify the bug
                identified_correct = any(
                    kw in text.lower()
                    for kw in ["bug", "error", "issue", "wrong", "问题", "错误", "bug", "缺陷", "漏"]
                )
                false_positive = False

            results.append({
                "name": t["name"],
                "severity": t["severity"],
                "has_bug": t["bug_description"] is not None,
                "identified_correctly": identified_correct,
                "false_positive": false_positive if t["bug_description"] is None else False,
                "response": text[:300],
            })
        except Exception as e:
            results.append({
                "name": t["name"], "error": str(e),
                "identified_correctly": False, "false_positive": False,
            })
    return results


def eval_reasoning(url, tests):
    """Evaluate multi-step reasoning accuracy."""
    results = []
    for t in tests:
        try:
            r = api_chat(url, [{"role": "user", "content": t["question"]}],
                         max_tokens=2048, temperature=0)
            text = extract_response(r["choices"][0]["message"])
            correct = t["check"](text)
            results.append({
                "name": t["name"], "question": t["question"],
                "answer": t["answer"], "response": text[:400],
                "correct": correct,
            })
        except Exception as e:
            results.append({
                "name": t["name"], "question": t["question"],
                "error": str(e), "correct": False,
            })
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Comparison
# ═══════════════════════════════════════════════════════════════════════════

def pearson_r(x, y):
    n = min(len(x), len(y))
    if n < 2:
        return float("nan")
    x, y = x[:n], y[:n]
    mx, my = sum(x) / n, sum(y) / n
    dx = [a - mx for a in x]
    dy = [b - my for b in y]
    num = sum(a * b for a, b in zip(dx, dy))
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    return num / den if den else float("nan")


def compare(cur, baseline_path):
    base = json.loads(Path(baseline_path).read_text())
    hdr = "=" * 70

    print(f"\n{hdr}")
    print("  QUALITY COMPARISON: Current vs Baseline")
    print(f"{hdr}")

    # ── 1. Logprob ─────────────────────────────────────────────────────
    cl = cur.get("logprob_eval", [])
    bl = base.get("logprob_eval", [])
    if cl and bl:
        print("\n── 1. Logprob Fidelity ──")
        exact = sum(1 for c, b in zip(cl, bl)
                    if "error" not in c and "error" not in b and c["generated"] == b["generated"])
        total = sum(1 for c, b in zip(cl, bl) if "error" not in c and "error" not in b)

        all_r = []
        for c, b in zip(cl, bl):
            if "error" in c or "error" in b:
                continue
            r = pearson_r(c["token_logprobs"], b["token_logprobs"])
            if not math.isnan(r):
                all_r.append(r)

        if total:
            print(f"  Greedy exact match:    {exact}/{total} ({100*exact/total:.1f}%)")
        if all_r:
            print(f"  Logprob Pearson r:     {sum(all_r)/len(all_r):.6f}")

        # Show mismatches
        if exact < total:
            print(f"\n  Mismatch details ({total - exact} prompts):")
            cnt = 0
            for c, b in zip(cl, bl):
                if "error" in c or "error" in b:
                    continue
                if c["generated"] != b["generated"]:
                    cnt += 1
                    print(f"    [{cnt}] '{c['prompt'][:40]}...'")
                    print(f"        base: '{b['generated'][:60]}'")
                    print(f"        cur:  '{c['generated'][:60]}'")
                    if cnt >= 5:
                        print("    ... (truncated)")
                        break

    # ── 2. QA Accuracy ─────────────────────────────────────────────────
    cq = cur.get("qa_eval", [])
    bq = base.get("qa_eval", [])
    if cq and bq:
        print("\n── 2. QA Accuracy ──")
        ca = sum(1 for r in cq if r["correct"])
        ba = sum(1 for r in bq if r["correct"])
        print(f"  Baseline: {ba}/{len(bq)} ({100*ba/len(bq):.1f}%)")
        print(f"  Current:  {ca}/{len(cq)} ({100*ca/len(cq):.1f}%)")
        print(f"  Delta:    {(ca - ba) * 100 / len(cq):+.1f}%")
        for c, b in zip(cq, bq):
            if b["correct"] and not c["correct"]:
                print(f"  REGRESSION: {c['question']}")
                print(f"    expected: {c['expected']}")
                print(f"    got: {c.get('response', c.get('error', ''))[:80]}")

    # ── 3. Tool Call ───────────────────────────────────────────────────
    ct = cur.get("tool_call_eval", [])
    bt = base.get("tool_call_eval", [])
    if ct and bt:
        print("\n── 3. Tool Call Accuracy ──")
        metrics = ["has_tool_call", "function_match", "json_parseable", "all_args_ok"]
        for m in metrics:
            cc = sum(1 for r in ct if r.get(m))
            bc = sum(1 for r in bt if r.get(m))
            delta = cc - bc
            sign = "+" if delta > 0 else ""
            print(f"  {m:20s}: baseline={bc}/{len(bt)}, current={cc}/{len(ct)} ({sign}{delta})")

    # ── 4. Bug Finder ─────────────────────────────────────────────────
    cb = cur.get("bug_finder_eval", [])
    bb = base.get("bug_finder_eval", [])
    if cb and bb:
        print("\n── 4. Bug Finding Accuracy ──")
        buggy = [(c, b) for c, b in zip(cb, bb) if c.get("has_bug")]
        clean = [(c, b) for c, b in zip(cb, bb) if not c.get("has_bug")]

        if buggy:
            bc_find = sum(1 for c, b in buggy if b.get("identified_correctly"))
            cc_find = sum(1 for c, b in buggy if c.get("identified_correctly"))
            print(f"  Bug detection:   baseline={bc_find}/{len(buggy)}, current={cc_find}/{len(buggy)}")
        if clean:
            bc_fp = sum(1 for c, b in clean if b.get("false_positive"))
            cc_fp = sum(1 for c, b in clean if c.get("false_positive"))
            print(f"  False positives: baseline={bc_fp}/{len(clean)}, current={cc_fp}/{len(clean)}")

    # ── 5. Reasoning ───────────────────────────────────────────────────
    cr = cur.get("reasoning_eval", [])
    br = base.get("reasoning_eval", [])
    if cr and br:
        print("\n── 5. Reasoning Accuracy ──")
        cc_r = sum(1 for r in cr if r["correct"])
        bc_r = sum(1 for r in br if r["correct"])
        print(f"  Baseline: {bc_r}/{len(br)} ({100*bc_r/len(br):.1f}%)")
        print(f"  Current:  {cc_r}/{len(cr)} ({100*cc_r/len(cr):.1f}%)")
        print(f"  Delta:    {(cc_r - bc_r) * 100 / len(cr):+.1f}%")
        for c, b in zip(cr, br):
            if b["correct"] and not c["correct"]:
                print(f"  REGRESSION: {c['name']}: {c['question'][:60]}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{hdr}")
    print("  SUMMARY")
    print(hdr)
    scores = {}
    if cq and bq:
        scores["QA"] = sum(1 for r in cq if r["correct"]) / len(cq)
    if ct and bt:
        scores["Tool Call"] = sum(1 for r in ct if r.get("function_match")) / len(ct)
    if cb and bb:
        scores["Bug Finder"] = sum(1 for r in cb if r.get("identified_correctly")) / len(cb)
    if cr and br:
        scores["Reasoning"] = sum(1 for r in cr if r["correct"]) / len(cr)
    if cl and bl:
        scores["Greedy Match"] = sum(
            1 for c, b in zip(cl, bl)
            if "error" not in c and "error" not in b and c["generated"] == b["generated"]
        ) / max(sum(1 for c, b in zip(cl, bl) if "error" not in c and "error" not in b), 1)

    for name, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {name:15s}: {score:6.1%} {bar}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Comprehensive model quality evaluation")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--baseline", default=None, help="Baseline JSON to compare")
    ap.add_argument("--skip-logprob", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--skip-tool-call", action="store_true")
    ap.add_argument("--skip-bug-finder", action="store_true")
    ap.add_argument("--skip-reasoning", action="store_true")
    args = ap.parse_args()

    results = {"url": args.url, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # Wait for server
    print("Waiting for server...", end=" ", flush=True)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{args.url}/v1/models", timeout=5)
            print("ready.")
            break
        except Exception:
            time.sleep(2)
    else:
        print("FAILED!", file=sys.stderr)
        sys.exit(1)

    if not args.skip_logprob:
        print("\n── 1/5 Logprob Evaluation ──")
        results["logprob_eval"] = eval_logprob(args.url, LOGPROB_PROMPTS)
        valid = [r for r in results["logprob_eval"] if "error" not in r]
        if valid:
            avg_lp = sum(sum(r["token_logprobs"]) / max(len(r["token_logprobs"]), 1) for r in valid) / len(valid)
            print(f"  {len(valid)} prompts, avg token logprob: {avg_lp:.4f}")

    if not args.skip_qa:
        print("\n── 2/5 QA Accuracy ──")
        results["qa_eval"] = eval_qa(args.url, QA_DATASET)
        ok = sum(1 for r in results["qa_eval"] if r["correct"])
        print(f"  {ok}/{len(QA_DATASET)} ({100*ok/len(QA_DATASET):.1f}%)")

    if not args.skip_tool_call:
        print("\n── 3/5 Tool Call ──")
        results["tool_call_eval"] = eval_tool_call(args.url, TOOL_CALL_TESTS)
        for r in results["tool_call_eval"]:
            status = "✓" if r.get("function_match") and r.get("all_args_ok") else "✗"
            print(f"  {status} {r['name']}: fn_match={r.get('function_match')}, args_ok={r.get('all_args_ok')}")

    if not args.skip_bug_finder:
        print("\n── 4/5 Bug Finder ──")
        results["bug_finder_eval"] = eval_bug_finder(args.url, BUG_FINDER_TESTS)
        ok = sum(1 for r in results["bug_finder_eval"] if r.get("identified_correctly"))
        fp = sum(1 for r in results["bug_finder_eval"] if r.get("false_positive"))
        print(f"  Identified: {ok}/{len(BUG_FINDER_TESTS)}, False positives: {fp}")

    if not args.skip_reasoning:
        print("\n── 5/5 Reasoning ──")
        results["reasoning_eval"] = eval_reasoning(args.url, REASONING_TESTS)
        ok = sum(1 for r in results["reasoning_eval"] if r["correct"])
        print(f"  {ok}/{len(REASONING_TESTS)} ({100*ok/len(REASONING_TESTS):.1f}%)")
        for r in results["reasoning_eval"]:
            status = "✓" if r["correct"] else "✗"
            if not r["correct"]:
                print(f"  {status} {r['name']}: got '{r.get('response', '')[:60]}'")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.output}")

    if args.baseline:
        compare(results, args.baseline)


if __name__ == "__main__":
    main()
