"""Ask a local LLM to classify each PSF test item; write one JSON line per item to results/raw/<run>.jsonl.

Backends
  chat       OpenAI-compatible /v1/chat/completions (Nemotron 8100, Kimi 8200, OpenJev helper 3000)
  systemone  OpenJev /v1/systemone "choice" question: one forward pass, probabilities for every level
  mock       random answers, for rehearsing the pipeline without the cluster

Keys are read from files (never from the command line): --key-file .tmp/llm-api-key etc.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

import httpx

from bench.reference import load_spec
from bench.spec import ROOT

SYSTEM = (
    "당신은 철도 분야 인간신뢰도분석(HRA) 전문가입니다. 작업자의 근무 조건을 보고 "
    "수행영향요인(PSF)의 수준을 판정합니다. 나쁜 수준일수록 인적 오류 확률(HEP)이 커집니다."
)


def level_text(levels: list[str]) -> str:
    return " < ".join(levels) + " (오른쪽일수록 오류 가능성이 큼)"


def conditions_text(psf: dict, choice: dict) -> str:
    return "\n".join(f"- {it['label']}: {it['options'][choice[it['key']]]['text']}" for it in psf["items"])


SHOTS_TITLE = "참고 사례(기존 예측 모델의 판정):"  # --shots-title overrides it (e.g. for expert labels)
HINT = ""  # --hint: extra sentence after the question (e.g. "answer what most workers would choose")


class Retriever:
    """Nearest training answers for a scenario (fewest differing items), train respondents only."""

    def __init__(self, k: int) -> None:
        split = json.loads((ROOT / "data/expert/split.json").read_text(encoding="utf-8"))
        train = set(split["train"])
        self.k = k
        self.rows: dict[int, list[dict]] = {}
        for r in map(json.loads, open(ROOT / "data/survey/responses.jsonl", encoding="utf-8")):
            if r["rid"] in train:
                self.rows.setdefault(r["psf"], []).append(r)

    def __call__(self, psf: dict, item: dict) -> tuple[list[dict], str]:
        pool = self.rows[item["psf"]]
        keys = [it["key"] for it in psf["items"]]
        rng = random.Random(item["id"])  # stable tie-break per question
        scored = sorted(((sum(r["choice"][k] != item["choice"][k] for k in keys), rng.random(), r) for r in pool),
                        key=lambda t: (t[0], t[1]))
        same = [r["answer"] for d, _, r in scored if d == 0]
        note = ""
        if same:
            counts = ", ".join(f"{lv} {same.count(lv)}" for lv in psf["levels"] if same.count(lv))
            note = f"\n판정할 조건과 똑같은 조건에 대한 학습 설문 응답 {len(same)}건: {counts}\n"
        shots = [{"choice": r["choice"], "label": r["answer"], "diff": d} for d, _, r in scored[: self.k]]
        shots.reverse()  # closest example last, right before the question
        return shots, note


def build_messages(psf: dict, item: dict, shots: list[dict], note: str = "") -> list[dict]:
    lv = psf["levels"]
    head = (f"PSF: {psf['name']}\n설문 문항: {psf['question']}\n가능한 수준: {level_text(lv)}\n")
    ex = ""
    if shots:
        ex = f"\n{SHOTS_TITLE}\n" + "\n\n".join(
            f"[사례 {i + 1}]" + (f" (다른 문항 {s['diff']}개)" if "diff" in s else "")
            + f"\n{conditions_text(psf, s['choice'])}\n=> {s['label']}" for i, s in enumerate(shots)) + "\n"
    ex += note
    ask = (f"{ex}\n판정할 조건:\n{conditions_text(psf, item['choice'])}\n\n"
           f"위 조건에서 '{psf['name']}' 수준을 {', '.join(lv)} 중 하나로 고르세요. {HINT}"
           f'다른 말 없이 JSON 한 줄로만 답하세요: {{"level": "<수준>"}}')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": head + ask}]


def parse_level(text: str, levels: list[str]) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r'"level"\s*:\s*"([^"]+)"', text)
    cand = (m.group(1) if m else text).strip()
    if cand in levels:
        return cand
    hits = [lv for lv in sorted(levels, key=len, reverse=True) if lv in cand]  # "매우나쁨" before "나쁨"
    return hits[0] if hits else None


async def ask_chat(cli, a, psf, item, shots):
    body = {"model": a.model, "messages": build_messages(psf, item, shots, item.get("_note", "")), "max_tokens": a.max_tokens,
            "temperature": a.temperature}
    if a.extra:
        body.update(json.loads(a.extra))
    r = await cli.post("/v1/chat/completions", json=body)
    r.raise_for_status()
    j = r.json()
    msg = j["choices"][0]["message"]
    text = msg.get("content") or ""
    return {"pred": parse_level(text, psf["levels"]), "text": text[-400:],
            "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or ""),
            "usage": j.get("usage")}


async def ask_systemone(cli, a, psf, item, shots):
    criteria = {lv: f"{psf['name']} 수준이 '{lv}'" for lv in psf["levels"]}
    instr = (f"{SYSTEM}\n설문 문항: {psf['question']}\n가능한 수준: {level_text(psf['levels'])}\n"
             f"아래 조건에서 '{psf['name']}' 수준은 무엇입니까?")
    if shots:
        instr += "\n참고 사례:\n" + "\n".join(
            f"{conditions_text(psf, s['choice'])} => {s['label']}" for s in shots)
    body = {"model": "openjev", "state": "근무 조건:\n" + conditions_text(psf, item["choice"]),
            "questions": {"level": {"type": "choice", "instructions": instr, "criteria": criteria}}}
    r = await cli.post("/v1/systemone", json=body)
    r.raise_for_status()
    j = r.json()
    ans = j["answers"]["level"]
    return {"pred": ans.get("choice"), "probabilities": ans.get("probabilities"),
            "confidence": ans.get("confidence"), "usage": j.get("usage")}


async def ask_mock(cli, a, psf, item, shots):
    await asyncio.sleep(0.001)
    return {"pred": random.choice(psf["levels"])}


async def run(a) -> None:
    spec = load_spec()
    items = [json.loads(x) for x in open(ROOT / "data" / a.items, encoding="utf-8")]
    if a.psf:
        items = [i for i in items if i["psf"] in a.psf]
    if a.limit:
        items = items[: a.limit]
    shots_all = [json.loads(x) for x in open(ROOT / "data" / a.shots_file, encoding="utf-8")]
    out = ROOT / a.out_dir / f"{a.run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(x)["id"] for x in open(out, encoding="utf-8") if '"error"' not in x}
    todo = [i for i in items if i["id"] not in done]
    key = Path(a.key_file).read_text(encoding="utf-8").strip() if a.key_file else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    fn = {"chat": ask_chat, "systemone": ask_systemone, "mock": ask_mock}[a.backend]
    retriever = Retriever(a.retrieval) if a.retrieval else None
    sem = asyncio.Semaphore(a.concurrency)
    t_start = time.time()
    n_ok = 0
    async with httpx.AsyncClient(base_url=a.base_url, headers=headers, timeout=a.timeout) as cli:
        with open(out, "a", encoding="utf-8") as f:
            async def one(it):
                nonlocal n_ok
                psf = spec[str(it["psf"])]
                shots = [s for s in shots_all if s["psf"] == it["psf"]] if a.shots else []
                if retriever:
                    shots, note = retriever(psf, it)
                    it = dict(it, _note=note)
                async with sem:
                    t0 = time.time()
                    rec = {"id": it["id"], "psf": it["psf"], "label": it["label"], "run": a.run}
                    for attempt in range(3):
                        try:
                            rec.update(await fn(cli, a, psf, it, shots))
                            rec.pop("error", None)
                            break
                        except Exception as e:  # keep going; failures are counted in the score
                            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                            await asyncio.sleep(2 * (attempt + 1))
                    rec["latency_s"] = round(time.time() - t0, 3)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += "error" not in rec
            await asyncio.gather(*(one(it) for it in todo))
    wall = time.time() - t_start
    meta = ROOT / a.out_dir / f"{a.run}.meta.json"
    prev = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {"wall_s": 0, "n": 0}
    meta.write_text(json.dumps({"run": a.run, "backend": a.backend, "model": a.model, "shots": a.shots,
                                "concurrency": a.concurrency, "extra": a.extra,
                                "wall_s": round(prev["wall_s"] + wall, 1), "n": prev["n"] + len(todo)},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{a.run}: {n_ok}/{len(todo)} ok in {wall:.1f}s -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name, e.g. nemotron-zeroshot")
    ap.add_argument("--backend", choices=["chat", "systemone", "mock"], default="chat")
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="nemotron-3.5-super-vl")
    ap.add_argument("--key-file", default="")
    ap.add_argument("--shots", action="store_true", help="add the few-shot examples of the same PSF")
    ap.add_argument("--extra", default="", help="JSON merged into the chat request body")
    ap.add_argument("--items", default="items.jsonl", help="file under data/")
    ap.add_argument("--shots-file", default="shots.jsonl", help="few-shot pool, file under data/")
    ap.add_argument("--shots-title", default="", help="heading printed above the few-shot examples")
    ap.add_argument("--out-dir", default="results/raw", help="where run files go, relative to the repo")
    ap.add_argument("--psf", type=int, nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--hint", default="", help="sentence appended to the question")
    ap.add_argument("--retrieval", type=int, default=0, help="K nearest training answers as examples")
    a = ap.parse_args()
    global SHOTS_TITLE, HINT
    if a.shots_title:
        SHOTS_TITLE = a.shots_title
    if a.hint:
        HINT = a.hint.strip() + " "
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
