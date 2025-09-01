# builder_utils.py
import json
from datetime import datetime

# ---------- Defaults ----------
DEFAULT_SYSTEM = (
    "You are a helpful radiology assistant. Here I have one image, and please answer me the "
    "question by first providing your step by step thoughts in natural language using the supported "
    "actions provided in the Python code APIs, and then output the corresponding consistent code that "
    "use the APIs provided step by step. Finally, you execute the code, show me the intermediate "
    "results and the final answer, and then give me the response."
)

DEFAULT_TEMPLATE = {
    "include_sections": ["APIs", "Question", "Thought", "Code",  "Answer"],  # per-example sections
    "apis_scope": "per",                        # "per" | "global"
    "show_system_in_preview": True,
    "show_global_apis_in_preview": True,        # only used when apis_scope == "global"
}

DEFAULT_APIS = '''# Example API doc
def classify_image(img_array: np.array) -> str:
    """classify the image into one of the 14 modalities and return the class"""

def read_rgb(img_path: str) -> np.array:
    """read the rgb image from the file path and return a 3d array"""
'''

DEFAULT_QUESTION = "What modality is this image?"
DEFAULT_CODE = '''def answer_code():
    cls = classify_image(img_array)
    return cls
'''
DEFAULT_ANSWER = "MRI"

# Superset of columns for the dataset preview
COLUMNS = ["#", "answer", "thought", "question", "code", "apis", "instruction"]

# ---------- Helpers ----------
def _truncate(s, max_len=120):
    s = (s or "").replace("\n", " ⏎ ")
    return s if len(s) <= max_len else s[:max_len - 1] + "…"

def _record_to_row(idx_one_based, record, max_len=120):
    return [
        idx_one_based,
        (record.get("answer") or ""),
        _truncate(record.get("thought") or "", max_len),
        _truncate(record.get("question") or "", max_len),
        _truncate(record.get("code") or "", max_len),
        _truncate(record.get("apis") or "", max_len),     # empty if scope was global
        _truncate(record.get("system") or "", max_len),
    ]

def dataset_rows(state, max_len=120):
    return [_record_to_row(i, rec, max_len=max_len) for i, rec in enumerate(state, start=1)]

def _resolve_default_answer(answer):
    """If user left Answer blank, use DEFAULT_ANSWER."""
    ans = (answer or "").strip()
    return DEFAULT_ANSWER if ans == "" else answer

def _validate_inputs_template(tmpl, system, global_apis, apis, question, code, answer):
    """
    Only 'System' is required. All other sections are optional even if enabled.
    """
    if not (system or "").strip():
        return False, "Please fill: System."
    return True, ""


def _build_text(tmpl, system, global_apis, apis, question, thought, code, answer):
    """Render a formatted few-shot example according to the template & scope.
       Arg order: (..., question, thought, code, answer)
    """
    sections = set((tmpl or {}).get("include_sections", []))
    scope = (tmpl or {}).get("apis_scope", "per")
    show_system = bool((tmpl or {}).get("show_system_in_preview", True))
    show_global_apis = bool((tmpl or {}).get("show_global_apis_in_preview", True))
    answer = _resolve_default_answer(answer)

    blocks = []
    idx = 1

    if show_system:
        blocks.append(f"{idx}. Instruction\n<SYSTEM>{(system or '').strip()}</SYSTEM>\n")
        idx += 1

    if scope == "global" and show_global_apis and (global_apis or "").strip():
        blocks.append(f"{idx}. Tool APIs (Global)\n<APIs>\n{(global_apis or '').rstrip()}\n</APIs>\n")
        idx += 1

    if scope == "per" and "APIs" in sections and (apis or "").strip():
        blocks.append(f"{idx}. Tool APIs\n<APIs>\n{(apis or '').rstrip()}\n</APIs>\n")
        idx += 1

    if "Question" in sections and (question or "").strip():
        blocks.append(f"{idx}. The VQA question:\n{(question or '').strip()}\n")
        idx += 1

    if "Thought" in sections and (thought or "").strip():
        blocks.append(f"{idx}. Thought\n<THOUGHT>\n{(thought or '').strip()}\n</THOUGHT>\n")
        idx += 1

    if "Code" in sections and (code or "").strip():
        blocks.append(f"{idx}. The generated code\n<CODE>\n{(code or '').rstrip()}\n</CODE>\n")
        idx += 1

    if "Answer" in sections:
        blocks.append(f"{idx}. Answer: {(answer or '').strip()}\n")
        idx += 1

    return "\n".join(blocks).strip()

def render_preview_with_template(tmpl, system, global_apis, apis, question, thought, code, answer):
    # Only keep fields that are enabled in the template
    sections = set((tmpl or {}).get("include_sections", []))
    scope = (tmpl or {}).get("apis_scope", "per")

    if scope != "global" and "APIs" not in sections:
        apis = ""
    if "Question" not in sections:
        question = ""
    if "Code" not in sections:
        code = ""
    if "Thought" not in sections:
        thought = ""
    if "Answer" not in sections:
        answer = ""

    ok, msg = _validate_inputs_template(tmpl, system, global_apis, apis, question, code, answer)
    if not ok:
        return msg, None

    # NOTE: _build_text expects (..., question, code, thought, answer)
    text = _build_text(tmpl, system, global_apis, apis, question, thought, code, answer)
    return text, text

def to_json_record_with_template(tmpl, system, global_apis, apis, question, thought, code, answer):
    """
    Record contains only enabled per-example sections, plus 'system' always.
    For global APIs scope, we also store a snapshot 'global_apis' on the record,
    so viewing later reproduces the same preview even if the template changes.
    """
    sections = set((tmpl or {}).get("include_sections", []))
    scope = (tmpl or {}).get("apis_scope", "per")
    answer = _resolve_default_answer(answer)

    rec = {
        "meta": {
            "included_sections": list(sections),
            "apis_scope": scope,
            "show_system_in_preview": bool((tmpl or {}).get("show_system_in_preview", True)),
            "show_global_apis_in_preview": bool((tmpl or {}).get("show_global_apis_in_preview", True)),
        },
        "system": system,
    }

    if scope == "global" and (global_apis or "").strip():
        rec["global_apis"] = global_apis
    elif "APIs" in sections and (apis or "").strip():
        rec["apis"] = apis

    if "Question" in sections and (question or "").strip():
        rec["question"] = question
    if "Thought" in sections and (thought or "").strip():
        rec["thought"] = thought
    if "Code" in sections and (code or "").strip():
        rec["code"] = code
    if "Answer" in sections:
        rec["answer"] = answer
    return rec

# 4) Add example wrapper
def add_example_and_summarize_with_template(state, tmpl, system, global_apis, apis, question, thought, code, answer):
    ok, msg = _validate_inputs_template(tmpl, system, global_apis, apis, question, code, answer)
    if not ok:
        return state, f"⚠️ {msg}", len(state), dataset_rows(state)
    rec = to_json_record_with_template(tmpl, system, global_apis, apis, question, thought, code, answer)
    new_state = list(state) + [rec]
    return new_state, "✅ Added example.", len(new_state), dataset_rows(new_state)


def _format_record_for_view(record):
    meta = record.get("meta") or {}
    included = set(meta.get("included_sections") or [])
    tmpl = {
        "include_sections": list(included),
        "apis_scope": meta.get("apis_scope", "per"),
        "show_system_in_preview": bool(meta.get("show_system_in_preview", True)),
        "show_global_apis_in_preview": bool(meta.get("show_global_apis_in_preview", True)),
    }
    return _build_text(
        tmpl=tmpl,
        system=record.get("system"),
        global_apis=record.get("global_apis"),
        apis=record.get("apis"),
        question=record.get("question"),
        thought=record.get("thought"),
        code=record.get("code"),
        answer=record.get("answer"),
    )

def get_example_detail(state, index_one_based):
    try:
        idx = int(float(index_one_based))
    except Exception:
        return "Please enter a valid integer index."
    if idx < 1 or idx > len(state):
        return f"Index out of range. Enter 1–{len(state) if state else 0}."
    record = state[idx - 1]
    return _format_record_for_view(record)

def delete_example(state, index_one_based):
    try:
        idx = int(float(index_one_based))
    except Exception:
        return state, "Please enter a valid integer index.", len(state)
    if idx < 1 or idx > len(state):
        return state, f"Index out of range. Enter 1–{len(state) if state else 0}.", len(state)
    new_state = state[: idx - 1] + state[idx :]
    return new_state, f"🗑️ Deleted example #{idx}.", len(new_state)

def delete_example_and_summarize(state, index_one_based):
    new_state, msg, count = delete_example(state, index_one_based)
    return new_state, msg, count, dataset_rows(new_state)

def export_jsonl_with_options(state, duplicate_system=True):
    """
    Export dataset to JSONL (per-line records).
    If duplicate_system=True, keep 'system' in each record; otherwise remove it.
    Global APIs snapshots (if present) remain on the records.
    """
    if not state:
        return None, "No examples to export yet."
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"./fewshot_{ts}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in state:
            out = dict(rec)
            if not duplicate_system:
                out.pop("system", None)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return path, f"📦 Exported {len(state)} records → {path}"

def export_single_json_object(state, system_text, template, global_apis_from_ui):
    """
    Export a single JSON object:
    {
      "system": <string>,
      "apis": <string?>,           # present when apis_scope == "global"
      "example": [ {apis?, question?, code?, thought?, answer?}, ... ]
    }

    Note: if current template uses global APIs, per-example 'apis' are omitted
    in the 'example' array; otherwise (per-example scope) they are included when present.
    """
    if not state:
        return None, "No examples to export yet."

    scope = (template or {}).get("apis_scope", "per")
    obj = {"system": system_text, "example": []}
    if scope == "global":
        obj["apis"] = global_apis_from_ui

    for rec in state:
        ex = {}
        # Include per-example APIs only when scope is "per"
        if scope == "per" and "apis" in rec:
            ex["apis"] = rec["apis"]
        if "question" in rec: ex["question"] = rec["question"]
        if "code" in rec:     ex["code"] = rec["code"]
        if "thought" in rec:  ex["thought"] = rec["thought"]
        if "answer" in rec:   ex["answer"] = rec["answer"]
        obj["example"].append(ex)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"./fewshot_object_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path, f"📦 Exported object with {len(obj['example'])} examples → {path}"
