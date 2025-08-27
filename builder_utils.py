# builder_utils.py
import json
from datetime import datetime

# ---------- Template & Defaults ----------

TEMPLATE_MARKED = """1. Instruction
<SYSTEM>{system}</SYSTEM>

2. Tool APIs
<APIs>
{apis}
</APIs>

3. The VQA question:
{question}

4. The generated code
<CODE>
{code}
</CODE>

5. Answer: {answer}
"""

DEFAULT_SYSTEM = (
    "You are a helpful radiology assistant. Here I have one image, and please answer me the "
    "question by first providing your step by step thoughts in natural language using the supported "
    "actions provided in the Python code APIs, and then output the corresponding consistant code that "
    "use the APIs provided step by step. Finally, you execute the code, show me the intermediate "
    "results and the final answer, and then give me the response."
)

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

DEFAULT_QUESTION = "What modality is this image? The image is in the path path/to/the/image"
DEFAULT_IMAGE_PATH = "path/to/the/image"

# NOTE: avoid reserved name `class`; use `cls`.
DEFAULT_CODE = '''def answer_code():
    img_array = read_rgb("path/to/the/image")
    cls = classify_image(img_array)
    return cls
'''

DEFAULT_ANSWER = "MRI"

# Column headers for the dataset table (truncated previews)
COLUMNS = ["#", "answer", "image_path", "question", "code", "apis", "instruction"]

# ---------- Core helpers ----------

def to_json_record(system, apis, question, code, answer, meta=None):
    """Compact JSON-friendly snapshot for dataset export."""
    return {
        "instruction": system,
        "apis": apis,
        "question": question,
        "code": code,
        "answer": answer,
        "meta": meta or {}
    }

def validate_inputs(system, apis, question, code, answer):
    """Return (ok, message)."""
    missing = []
    if not system or not system.strip(): missing.append("Instruction (SYSTEM)")
    if not apis or not apis.strip(): missing.append("Tool APIs")
    if not question or not question.strip(): missing.append("VQA question")
    if not code or not code.strip(): missing.append("Generated code")
    if not answer or not answer.strip(): missing.append("Answer")
    if missing:
        return False, f"Please fill: {', '.join(missing)}."
    return True, ""

def _truncate(s, max_len=120):
    s = (s or "").replace("\n", " ⏎ ")
    return s if len(s) <= max_len else s[:max_len - 1] + "…"

def _record_to_row(idx_one_based, record, max_len=120):
    meta = record.get("meta") or {}
    return [
        idx_one_based,
        record.get("answer", ""),
        meta.get("image_path", ""),
        _truncate(record.get("question", ""), max_len),
        _truncate(record.get("code", ""), max_len),
        _truncate(record.get("apis", ""), max_len),
        _truncate(record.get("instruction", ""), max_len),
    ]

def dataset_rows(state, max_len=120):
    """Return 2D list (rows) for the dataset table."""
    rows = []
    for i, rec in enumerate(state, start=1):
        rows.append(_record_to_row(i, rec, max_len=max_len))
    return rows

def format_record(record):
    """Full, nicely formatted example text for viewing."""
    return TEMPLATE_MARKED.format(
        system=(record.get("instruction") or "").strip(),
        apis=(record.get("apis") or "").rstrip(),
        question=(record.get("question") or "").strip(),
        code=(record.get("code") or "").rstrip(),
        answer=(record.get("answer") or "").strip(),
    )

def render_preview(system, apis, question, image_path, code, answer):
    """
    Build the formatted example text and return (display_text, raw_text_for_copy).
    Replaces 'path/to/the/image' inside the question with the provided image_path.
    """
    if image_path and "path/to/the/image" in (question or ""):
        question = question.replace("path/to/the/image", image_path)

    ok, msg = validate_inputs(system, apis, question, code, answer)
    if not ok:
        return msg, None

    text = TEMPLATE_MARKED.format(
        system=system.strip(),
        apis=apis.rstrip(),
        question=question.strip(),
        code=code.rstrip(),
        answer=answer.strip()
    )
    return text, text

def add_example(state, system, apis, question, image_path, code, answer):
    """
    Append current example to dataset state. Returns (new_state, feedback_msg, count).
    The `state` is a list of JSON records.
    """
    text, _ = render_preview(system, apis, question, image_path, code, answer)
    if not isinstance(text, str) or not text.startswith("1. Instruction"):
        return state, f"⚠️ {text}", len(state)

    record = to_json_record(system, apis, question, code, answer, meta={"image_path": image_path})
    new_state = list(state) + [record]
    return new_state, "✅ Added example.", len(new_state)

def add_example_and_summarize(state, system, apis, question, image_path, code, answer):
    """
    Add example, then return rows for visualization.
    Returns (new_state, message, count, rows)
    """
    new_state, msg, count = add_example(state, system, apis, question, image_path, code, answer)
    rows = dataset_rows(new_state)
    return new_state, msg, count, rows

def get_example_detail(state, index_one_based):
    """
    Return the full formatted text of the selected example (1-based index).
    If out of range, return a friendly message.
    """
    try:
        idx = int(index_one_based)
    except Exception:
        return "Please enter a valid integer index."
    if idx < 1 or idx > len(state):
        return f"Index out of range. Enter 1–{len(state) if state else 0}."
    record = state[idx - 1]
    return format_record(record)

# -------- NEW: delete helpers --------

def delete_example(state, index_one_based):
    """
    Delete the example at 1-based index.
    Returns (new_state, message, count)
    """
    try:
        idx = int(index_one_based)
    except Exception:
        return state, "Please enter a valid integer index.", len(state)
    if idx < 1 or idx > len(state):
        return state, f"Index out of range. Enter 1–{len(state) if state else 0}.", len(state)
    new_state = state[: idx - 1] + state[idx :]
    return new_state, f"🗑️ Deleted example #{idx}.", len(new_state)

def delete_example_and_summarize(state, index_one_based):
    """
    Delete example, then return rows for visualization.
    Returns (new_state, message, count, rows)
    """
    new_state, msg, count = delete_example(state, index_one_based)
    rows = dataset_rows(new_state)
    return new_state, msg, count, rows

def export_jsonl(state):
    """
    Export the dataset state to a timestamped JSONL file.
    Returns (filepath, status_message). If no data, returns (None, message).
    """
    if not state:
        return None, "No examples to export yet."
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"./fewshot_{ts}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in state:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path, f"📦 Exported {len(state)} examples → {path}"
