"""Extract the survey (PSF questions and answer options) from Railway_hep's input.html.

Each PSF block ("house") holds selects in the order F.p.S.1, F.p.S.2, ...; option i of select k
maps to the answer code F.p.S.k.(i+1), whose 0/1/2 value comes from config/mappings.code_to_value.
"""
from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "Railway_hep"

# CatBoost output labels and HEP multipliers, copied from Railway_hep result.html / config/mappings.py
LEVELS = {
    1: ["보통", "나쁨", "매우나쁨"],
    2: ["좋음", "보통", "나쁨"],
    3: ["좋음", "보통", "나쁨"],
    4: ["좋음", "보통", "나쁨"],
    5: ["보통", "나쁨"],
    6: ["보통", "나쁨"],
    7: ["좋음", "보통", "나쁨"],
    8: ["보통", "약간 높음", "매우 높음"],
}
MULTIPLIERS = {
    1: {"보통": 1, "나쁨": 14.45, "매우나쁨": 17.54},
    2: {"좋음": 0.492, "보통": 1, "나쁨": 5.654},
    3: {"좋음": 0.292, "보통": 1, "나쁨": 8.559},
    4: {"좋음": 0.421, "보통": 1, "나쁨": 7.229},
    5: {"보통": 1, "나쁨": 9.559},
    6: {"보통": 1, "나쁨": 11.576},
    7: {"좋음": 0.322, "보통": 1, "나쁨": 12.11},
    8: {"보통": 1, "약간 높음": 4.654, "매우 높음": 11.873},
}
ROLE_BASE_HEP = {
    "기관사": (0.01225, 0.00615),
    "관제사": (0.0055, 0.00352),
    "정비사": (0.00279, 0.00347),
    "선로작업자": (0.01225, 0.0035),
}
PSF_NAMES = {1: "작업부하", 2: "장비 및 기기", 3: "절차 및 지침", 4: "교육 및 훈련",
             5: "의사소통", 6: "환경", 7: "작업절차", 8: "복잡성"}


# input.html lists these options in a different order than code_to_value assumes, so the web app sends the
# reverse value. CatBoost itself treats 2 as the worst level for every item (see docs/report.md), and
# config/mappings.py text maps agree with the values below.
SEMANTIC_FIX = {
    "F.1.S.3.1": 0, "F.1.S.3.3": 2,      # sleep: 8h or more -> 0, under 6h -> 2
    "F.3.S.2.1": 0, "F.3.S.2.3": 2,      # procedure complexity: low -> 0, high -> 2
    "F.6.S.9.1": 0, "F.6.S.9.2": 2,      # floor friction: SCOF >= 0.63 (safe) -> 0
    "F.6.S.7.1": 1, "F.6.S.7.2": 2,      # yellow dust: breathing discomfort -> 1, cannot see -> 2
}


class _SurveyParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.houses: list[dict] = []
        self._div_stack: list[str] = []
        self._in_title = self._in_label_text = self._in_option = False
        self._label = ""
        self._select: dict | None = None
        self._opt: dict | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "div":
            self._div_stack.append("house" if "house" in cls else "title" if "house-title" in cls else "")
            if "house" in cls:
                self.houses.append({"title": "", "selects": []})
            if "house-title" in cls:
                self._in_title = True
        elif tag == "span" and "label-text" in cls:
            self._in_label_text, self._label = True, ""
        elif tag == "select" and "room-select" in cls and self.houses:
            self._select = {"label": self._label.strip(), "options": []}
        elif tag == "option" and self._select is not None:
            disabled = "disabled" in a or a.get("value", "") == ""
            self._opt = None if disabled else {"text": ""}
            self._in_option = True

    def handle_endtag(self, tag):
        if tag == "div" and self._div_stack:
            if self._div_stack.pop() == "title":
                self._in_title = False
        elif tag == "span":
            self._in_label_text = False
        elif tag == "option":
            if self._opt is not None and self._select is not None:
                self._select["options"].append(self._opt["text"].strip())
            self._opt, self._in_option = None, False
        elif tag == "select" and self._select is not None:
            self.houses[-1]["selects"].append(self._select)
            self._select = None

    def handle_data(self, data):
        if self._in_title and self.houses:
            self.houses[-1]["title"] += data
        if self._in_label_text:
            self._label += data
        if self._in_option and self._opt is not None:
            self._opt["text"] += data


def option_code_table(html: str) -> dict[str, list[int]]:
    """input.html's optionCodeNumbersByField: screen option index -> code number (empty in f864cce)."""
    m = re.search(r"const optionCodeNumbersByField\s*=\s*\{(.*?)\};", html, re.S)
    if not m:
        return {}
    return {k: [int(x) for x in v.split(",")]
            for k, v in re.findall(r"'(F\.\d\.S\.\d+)'\s*:\s*\[([\d,\s]+)\]", m.group(1))}


def build_spec(vendor: Path = VENDOR) -> dict:
    sys.path.insert(0, str(vendor))
    from config.mappings import code_to_value  # noqa: E402

    html = (vendor / "input.html").read_text(encoding="utf-8-sig")
    table = option_code_table(html)
    p = _SurveyParser()
    p.feed(html)
    houses = [h for h in p.houses if h["selects"]]
    if len(houses) != 8:
        raise RuntimeError(f"expected 8 PSF blocks, found {len(houses)}")
    spec = {}
    for pid, h in enumerate(houses, start=1):
        items = []
        for k, sel in enumerate(h["selects"], start=1):
            opts = []
            key = f"F.{pid}.S.{k}"
            for i, text in enumerate(sel["options"], start=1):
                code = f"F.{pid}.S.{k}.{i}"
                if code not in code_to_value:
                    raise KeyError(code)
                sent = f"{key}.{table[key][i - 1]}" if key in table else code  # what getValue() sends
                web = int(code_to_value[sent])
                fixed = SEMANTIC_FIX.get(code, web)
                # value: encoding the CatBoost model was trained with (2 = worst); value_web: what input.html sends
                opts.append({"code": code, "text": text, "value": fixed, "value_web": web})
            items.append({"key": f"F.{pid}.S.{k}", "label": sel["label"], "options": opts})
        spec[str(pid)] = {"name": PSF_NAMES[pid], "question": " ".join(h["title"].split()),
                          "levels": LEVELS[pid], "items": items}
    return spec


if __name__ == "__main__":
    out = ROOT / "data" / "survey_spec.json"
    spec = build_spec()
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    for pid, s in spec.items():
        print(pid, s["name"], len(s["items"]), "items", [len(i["options"]) for i in s["items"]])
