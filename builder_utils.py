# builder_utils.py
import json
from datetime import datetime

# ---------- Defaults ----------
DEFAULT_SYSTEM = (
    "You are a helpful radiology assistant. Here I have one image, and please answer me the "
    "question by first providing your step by step thoughts in natural language using the supported "
    "actions provided in the Python code APIs, and then output the corresponding consistant code that "
    "use the APIs provided step by step. Finally, you execute the code, show me the intermediate "
    "results and the final answer, and then give me the response."
)

DEFAULT_TEMPLATE = {
    "include_sections": ["APIs", "Question", "Code", "Answer"],
    "show_system_in_preview": True,
}

DEFAULT_APIS = '''# something in style API doc in python
def classify_image(img_array: np.array) -> str:
    """classify the image into one of the 14 modalities and return the class
    Args:
        img_array: np.array, the np array of the image, in 3 dimension
    Return:
        res: str, the class of the image
    """

def read_rgb(img_path: str) -> np.array:
    """read the rgb image from the file path and return a 3d array
    Args:
        img_path: str the path to the file
    Return:
        img_array: the image array in 3 dimension
    """
'''

DEFAULT_QUESTION = "What modality is this image?"
DEFAULT_CODE = '''def answer_code():
    img_array = read_rgb("path/to/the/image")
    cls = classify_image(img_array)
    return cls
'''
DEFAULT_ANSWER = "MRI"

# Superset of columns for preview
COLUMNS = ["#", "answer", "question", "code", "apis", "instruction"]

# ---------- Helpers ----------
def _truncate(s, max_len=120):
    s = (s or "").replace("\n", " ⏎ ")
    return s if len(s) <= max_len else s[:max_len - 1] + "…"

def _record_to_row(idx_one_based, record, max_len=120):
    return [
        idx_one_based,
        record.get("answer", ""),
        _truncate(record.get("question", ""), max_len),
        _truncate(record.get("code", ""), max_len),
        _truncate(record.get("apis", ""), max_len),
        _truncate(record.get("system", ""), max_len),
    ]

def dataset_rows(state, max_len=120):
    return [_record_to_row(i, rec, max_len=max_len) for i, rec in enumerate(state, start=1)]

def _resolve_default_answer(answer):
    """If user left Answer blank, use DEFAULT_ANSWER."""
    ans = (answer or "").strip()
    return DEFAULT_ANSWER if ans == "" else answer

def _validate_inputs_template(tmpl, system, apis, question, code, answer):
    """
    Validate required fields based on template. 'Answer' may be blank,
    as it will resolve to DEFAULT_ANSWER.
    """
    missing = []
    if not (system or "").strip():
        missing.append("System")
    sections = set((tmpl or {}).get("include_sections", []))
    if "APIs" in sections and not (apis or "").strip():
        missing.append("APIs")
    if "Question" in sections and not (question or "").strip():
        missing.append("Question")
    if "Code" in sections and not (code or "").strip():
        missing.append("Code")
    # Answer is optional at input time (blank -> default)
    if missing:
        return False, f"Please fill: {', '.join(missing)}."
    return True, ""

def _build_text(tmpl, system, apis, question, code, answer):
    """Render a formatted few-shot example according to the template."""
    sections = set((tmpl or {}).get("include_sections", []))
    show_system = bool((tmpl or {}).get("show_system_in_preview", True))
    answer = _resolve_default_answer(answer)

    blocks = []
    idx = 1

    if show_system:
        blocks.append(f"{idx}. Instruction\n<SYSTEM>{(system or '').strip()}</SYSTEM>\n")
        idx += 1

    if "APIs" in sections:
        blocks.append(f"{idx}. Tool APIs\n<APIs>\n{(apis or '').rstrip()}\n</APIs>\n")
        idx += 1

    if "Question" in sections:
        blocks.append(f"{idx}. The VQA question:\n{(question or '').strip()}\n")
        idx += 1

    if "Code" in sections:
        blocks.append(f"{idx}. The generated code\n<CODE>\n{(code or '').rstrip()}\n</CODE>\n")
        idx += 1

    if "Answer" in sections:
        blocks.append(f"{idx}. Answer: {(answer or '').strip()}\n")
        idx += 1

    return "\n".join(blocks).strip()

def render_preview_with_template(tmpl, system, apis, question, code, answer):
    ok, msg = _validate_inputs_template(tmpl, system, apis, question, code, answer)
    if not ok:
        return msg, None
    text = _build_text(tmpl, system, apis, question, code, answer)
    return text, text

def to_json_record_with_template(tmpl, system, apis, question, code, answer):
    """Record contains only enabled sections. System stored too."""
    sections = set((tmpl or {}).get("include_sections", []))
    answer = _resolve_default_answer(answer)
    rec = {"meta": {
        "included_sections": list(sections),
        "show_system_in_preview": bool((tmpl or {}).get("show_system_in_preview", True)),
    }}
    rec["system"] = system
    if "APIs" in sections:
        rec["apis"] = apis
    if "Question" in sections:
        rec["question"] = question
    if "Code" in sections:
        rec["code"] = code
    if "Answer" in sections:
        rec["answer"] = answer
    return rec

def add_example_and_summarize_with_template(state, tmpl, system, apis, question, code, answer):
    ok, msg = _validate_inputs_template(tmpl, system, apis, question, code, answer)
    if not ok:
        return state, f"⚠️ {msg}", len(state), dataset_rows(state)
    rec = to_json_record_with_template(tmpl, system, apis, question, code, answer)
    new_state = list(state) + [rec]
    return new_state, "✅ Added example.", len(new_state), dataset_rows(new_state)

def _format_record_for_view(record):
    meta = record.get("meta") or {}
    included = set(meta.get("included_sections") or [])
    tmpl = {
        "include_sections": list(included),
        "show_system_in_preview": bool(meta.get("show_system_in_preview", True)),
    }
    return _build_text(
        tmpl=tmpl,
        system=record.get("system"),
        apis=record.get("apis"),
        question=record.get("question"),
        code=record.get("code"),
        answer=record.get("answer"),
    )

def get_example_detail(state, index_one_based):
    try:
        idx = int(index_one_based)
    except Exception:
        return "Please enter a valid integer index."
    if idx < 1 or idx > len(state):
        return f"Index out of range. Enter 1–{len(state) if state else 0}."
    record = state[idx - 1]
    return _format_record_for_view(record)

# -------- delete helpers --------
def delete_example(state, index_one_based):
    try:
        idx = int(index_one_based)
    except Exception:
        return state, "Please enter a valid integer index.", len(state)
    if idx < 1 or idx > len(state):
        return state, f"Index out of range. Enter 1–{len(state) if state else 0}.", len(state)
    new_state = state[: idx - 1] + state[idx :]
    return new_state, f"🗑️ Deleted example #{idx}.", len(new_state)

def delete_example_and_summarize(state, index_one_based):
    new_state, msg, count = delete_example(state, index_one_based)
    return new_state, msg, count, dataset_rows(new_state)

# -------- export --------
def export_jsonl_with_options(state, duplicate_system=True):
    """
    Export dataset to JSONL (per-line records).
    If duplicate_system=True, keep 'system' in each record; otherwise remove it.
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

def export_single_json_object(state, system_text):
    """
    Export a single JSON object:
    {
      "system": <string>,
      "example": [ {apis?, question?, code?, answer?}, ... ]
    }
    """
    if not state:
        return None, "No examples to export yet."

    obj = {"system": system_text, "example": []}
    for rec in state:
        ex = {}
        # Only include sections that exist in the saved record
        if "apis" in rec:     ex["apis"] = rec["apis"]
        if "question" in rec: ex["question"] = rec["question"]
        if "code" in rec:     ex["code"] = rec["code"]
        if "answer" in rec:   ex["answer"] = rec["answer"]
        obj["example"].append(ex)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"./fewshot_object_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path, f"📦 Exported object with {len(obj['example'])} examples → {path}"
