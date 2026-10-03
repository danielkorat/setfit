"""The five Jev-chart tasks: test sets (via upstream jsb.datasets loaders), unlabeled pools, and Jev's question text.

Run with upstream's venv (`upstream/.venv/bin/python`), which has `jsb` installed by `make install`.
Test cases and gold labels come only from upstream's loaders; pools are read from files in data/pools/
(see fetch_pools.py) and never overlap the test sets.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
DATA = UPSTREAM / "data"  # upstream `make download-data` output (WildGuardTest pinned to the walledai mirror)
POOLS = HERE / "data" / "pools"
sys.path.insert(0, str(UPSTREAM))

from jsb.datasets import aegis_v2, harmbench, toxicchat, wildguardtest  # noqa: E402
from jsb.policy.questions import POLICY_DIR, build_questions  # noqa: E402

LOADERS = {"aegis_v2": aegis_v2.load, "toxicchat": toxicchat.load, "wildguardtest": wildguardtest.load,
           "harmbench": harmbench.load}


@dataclass(frozen=True)
class Task:
    id: str
    benchmark: str
    label: str  # chart row label
    response: bool  # True = the classifier sees the response (and its prompt)
    published_n: int


# chart row order
TASKS = [
    Task("wg_prompt_harmful", "wildguardtest", "WildGuardTest prompt", False, 1_699),
    Task("ag2_unsafe_prompt", "aegis_v2", "Aegis v2 prompt", False, 1_928),
    Task("tc_toxic", "toxicchat", "ToxicChat prompt", False, 2_853),
    Task("hb_harmful_response", "harmbench", "HarmBench response", True, 596),
    Task("ag2_unsafe_response", "aegis_v2", "Aegis v2 response", True, 852),
]
BY_ID = {t.id: t for t in TASKS}


@dataclass(frozen=True)
class Item:
    id: str
    prompt: str
    response: str | None
    gold: bool | None  # None = no gold (never used for selection)


def input_text(task: Task, item: Item) -> str:
    """What SetFit sees: the prompt for prompt tasks, response then prompt for response tasks."""
    if task.response:
        return f"Response: {item.response}\n\nPrompt: {item.prompt}"
    return item.prompt


def load_test(task: Task) -> list[Item]:
    """Assessable test cases + gold for `task`, through upstream's loader; asserts the published n."""
    LOADERS[task.benchmark](DATA, None)  # idempotent: re-verifies the pinned shas, rewrites <bench>_cases.jsonl
    items = []
    for line in open(DATA / f"{task.benchmark}_cases.jsonl"):
        row = json.loads(line)
        if task.id not in row["tasks"] or row["gold"].get(task.id) is None:
            continue
        items.append(Item(row["case_id"], row["text"], row["response"], row["gold"][task.id]))
    assert len(items) == task.published_n, f"{task.id}: n={len(items)}, published {task.published_n}"
    assert len({i.id for i in items}) == len(items)
    return items


def question(task: Task) -> dict:
    """The exact question Jev received for this task: instructions + true/false criteria + benchmark policy text."""
    q = build_questions(task.benchmark)[task.id]
    return dict(instructions=q.instructions, criteria=dict(q.criteria),
                policy=(POLICY_DIR / f"{task.benchmark}_policy.md").read_text(encoding="utf-8"))


def _aegis_train() -> list[dict]:
    rows = json.load(open(POOLS / "nvidia__train.json"))
    return [r for r in rows if r["prompt"] != "REDACTED"]


def load_pool(task: Task, test: list[Item]) -> list[Item]:
    """Unlabeled pool for `task`, disjoint from its test set (exact-text overlaps removed)."""
    test_prompts = {i.prompt.strip() for i in test}
    test_responses = {(i.response or "").strip() for i in test if i.response}
    if task.id == "ag2_unsafe_prompt":
        seen, pool = set(), []
        for r in _aegis_train():
            p = r["prompt"].strip()
            if p in test_prompts or p in seen:
                continue
            seen.add(p)
            pool.append(Item(f"ag2tr_{r['id']}", r["prompt"], None, {"unsafe": True, "safe": False}.get(r["prompt_label"])))
        return pool
    if task.id == "ag2_unsafe_response":
        return [Item(f"ag2tr_{r['id']}", r["prompt"], r["response"], {"unsafe": True, "safe": False}.get(r["response_label"]))
                for r in _aegis_train()
                if r.get("response") and r["response"] != "REDACTED" and r["response"].strip() not in test_responses]
    if task.id == "tc_toxic":
        pool = []
        for r in csv.DictReader(open(POOLS / "lmsys__toxic-chat_annotation_train.csv", encoding="utf-8")):
            # same human_annotation filter upstream applies to the test split (metadata, not the label)
            if r["human_annotation"] != "True" or r["user_input"].strip() in test_prompts:
                continue
            pool.append(Item(f"tctr_{r['conv_id']}", r["user_input"], None, r["toxicity"] == "1"))
        return pool
    if task.id == "wg_prompt_harmful":
        import pyarrow.parquet as pq

        seen, pool = set(), []
        for r in pq.read_table(POOLS / "allenai__wildguard_train.parquet", columns=["prompt", "prompt_harm_label"]).to_pylist():
            p = (r["prompt"] or "").strip()
            if not p or p in test_prompts or p in seen:
                continue
            seen.add(p)
            pool.append(Item(f"wgtr_{len(pool)}", r["prompt"], None, {"harmful": True, "unharmful": False}.get(r["prompt_harm_label"])))
        return pool
    if task.id == "hb_harmful_response":
        return list(test)  # the 596 test items; selected ones are excluded from scoring
    raise KeyError(task.id)


if __name__ == "__main__":
    for t in TASKS:
        test = load_test(t)
        pool = load_pool(t, test)
        pos = sum(i.gold for i in test)
        print(f"{t.id}: test n={len(test)} (pos {pos}) pool n={len(pool)} "
              f"pool gold pos={sum(1 for i in pool if i.gold)} none={sum(i.gold is None for i in pool)}")
