# app.py
import gradio as gr
from builder_utils import (
    DEFAULT_SYSTEM,
    DEFAULT_TEMPLATE,
    DEFAULT_APIS,
    DEFAULT_QUESTION,
    DEFAULT_CODE,
    DEFAULT_ANSWER,
    COLUMNS,
    render_preview_with_template,
    add_example_and_summarize_with_template,
    get_example_detail,
    delete_example_and_summarize,
    export_jsonl_with_options,
    export_single_json_object,
)

def build_app():
    with gr.Blocks(title="Few-shot Prompt Builder — Custom Template", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## 🧪 Few-shot Prompt Builder — Custom Template\n"
            "Use a single System message and choose which sections your few-shot examples include."
        )

        # States
        dataset_state = gr.State([])              # list[dict]
        template_state = gr.State(DEFAULT_TEMPLATE.copy())
        keep_system_state = gr.State(True)        # hidden bool for JSONL export

        with gr.Tabs():
            # -------------------- Template Tab --------------------
            with gr.TabItem("Template"):
                with gr.Row():
                    with gr.Column(scale=1):
                        system_global = gr.Textbox(
                            label="System message (global)",
                            value=DEFAULT_SYSTEM,
                            lines=6,
                            placeholder="Your SYSTEM prompt here..."
                        )
                        include_sections = gr.CheckboxGroup(
                            label="Sections to include in each example",
                            choices=["APIs", "Question", "Code", "Answer"],
                            value=DEFAULT_TEMPLATE["include_sections"],
                        )
                        show_system_in_preview = gr.Checkbox(
                            label="Show System at top of previews",
                            value=DEFAULT_TEMPLATE["show_system_in_preview"]
                        )
                        apply_btn = gr.Button("✅ Apply template settings", variant="primary")
                        template_feedback = gr.Textbox(label="Template status", interactive=False)

                    with gr.Column(scale=1):
                        gr.Markdown(
                            "### How it works\n"
                            "- Toggle which sections **exist** in each example\n"
                            "- The **Single Example** tab will show/hide inputs accordingly"
                        )

            # -------------------- Single Example Tab --------------------
            with gr.TabItem("Single Example"):
                with gr.Row():
                    with gr.Column(scale=1):
                        apis = gr.Code(
                            label="APIs (Python doc / signatures)",
                            value=DEFAULT_APIS,
                            language="python",
                            lines=14,
                            visible=("APIs" in DEFAULT_TEMPLATE["include_sections"])
                        )
                        question = gr.Textbox(
                            label="Question",
                            value=DEFAULT_QUESTION,
                            lines=2,
                            visible=("Question" in DEFAULT_TEMPLATE["include_sections"])
                        )
                        code = gr.Code(
                            label="Code",
                            value=DEFAULT_CODE,
                            language="python",
                            lines=10,
                            visible=("Code" in DEFAULT_TEMPLATE["include_sections"])
                        )
                        # Smart default Answer: empty means use DEFAULT_ANSWER
                        answer = gr.Textbox(
                            label="Answer",
                            value="",
                            placeholder=DEFAULT_ANSWER,
                            lines=1,
                            visible=("Answer" in DEFAULT_TEMPLATE["include_sections"])
                        )

                        with gr.Row():
                            preview_btn = gr.Button("🔎 Preview Example", variant="secondary")
                            add_btn = gr.Button("➕ Add Current Example to Dataset", variant="primary")
                        feedback_single = gr.Textbox(label="Single Example status", lines=2, interactive=False)

                    with gr.Column(scale=1):
                        preview = gr.Textbox(label="Formatted Example Preview", lines=32)
                        copy_helper = gr.Textbox(label="(Hidden) Raw", visible=False)
                        gr.Markdown("Tip: copy from the preview box after rendering.")

                preview_btn.click(
                    render_preview_with_template,
                    inputs=[template_state, system_global, apis, question, code, answer],
                    outputs=[preview, copy_helper]
                )

            # -------------------- Dataset Builder Tab --------------------
            with gr.TabItem("Dataset Builder"):
                with gr.Row():
                    with gr.Column(scale=1):
                        count = gr.Number(
                            label="Examples in dataset",
                            value=0,
                            precision=0,
                            interactive=False,
                        )
                        dataset_table = gr.Dataframe(
                            headers=COLUMNS,
                            value=[],
                            wrap=True,
                            interactive=False,
                            label="Dataset (truncated previews)"
                        )
                        with gr.Row():
                            view_index = gr.Number(  # 1-based index
                                label="Select example # (1-based)",
                                value=1,
                                precision=0,
                                minimum=1,
                                step=1,
                            )
                            view_btn = gr.Button("👁️ View Selected")
                            delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")
                        builder_feedback = gr.Textbox(label="Dataset status", lines=2, interactive=False)
                    with gr.Column(scale=1):
                        full_view = gr.Textbox(label="Selected Example (full)", lines=34)

                # ---------- Helpers ----------
                def _sanitize_index(idx, total):
                    """Coerce to int in [1, total] (or 1 if total==0)."""
                    try:
                        i = int(float(idx))
                    except Exception:
                        i = 1
                    if total <= 0:
                        return 1
                    return max(1, min(i, total))

                # On number change (typing or arrow keys): coerce to int and update viewer
                def _on_index_change(idx, state):
                    total = len(state or [])
                    clean = _sanitize_index(idx, total)
                    view = get_example_detail(state, clean) if total else "Dataset is empty."
                    return clean, view

                view_index.change(
                    _on_index_change,
                    inputs=[view_index, dataset_state],
                    outputs=[view_index, full_view],
                )

                # View button: clamp index, update viewer and write back the cleaned index
                def _view_and_fix(state, idx):
                    total = len(state or [])
                    idx1 = _sanitize_index(idx, total)
                    return get_example_detail(state, idx1), idx1

                view_btn.click(
                    _view_and_fix,
                    inputs=[dataset_state, view_index],
                    outputs=[full_view, view_index]
                )

                # Delete: refresh viewer & clamp index
                def _delete_and_refresh(state, idx):
                    new_state, msg, new_count, rows = delete_example_and_summarize(state, idx)
                    if new_count == 0:
                        return new_state, msg, new_count, rows, "Dataset is now empty.", 1
                    idx1 = _sanitize_index(idx, new_count)
                    full = get_example_detail(new_state, idx1)
                    return new_state, msg, new_count, rows, full, idx1

                delete_btn.click(
                    _delete_and_refresh,
                    inputs=[dataset_state, view_index],
                    outputs=[dataset_state, builder_feedback, count, dataset_table, full_view, view_index]
                )

                with gr.Row():
                    export_btn = gr.Button("📤 Export JSONL (per-line records)")
                    export_file = gr.File(label="Download JSONL", interactive=False)
                    export_status = gr.Textbox(label="Export status", lines=2)
                export_btn.click(
                    export_jsonl_with_options,
                    inputs=[dataset_state, keep_system_state],
                    outputs=[export_file, export_status]
                )

                with gr.Row():
                    export_json_btn = gr.Button("📦 Export JSON (single object)")
                    export_json_file = gr.File(label="Download JSON", interactive=False)
                    export_json_status = gr.Textbox(label="Export status", lines=2)
                export_json_btn.click(
                    export_single_json_object,
                    inputs=[dataset_state, system_global],
                    outputs=[export_json_file, export_json_status]
                )

        # ---------- Template apply wiring (toggles Single Example visibility)
        def _apply_template(includes, show_sys, tmpl_state):
            if not includes:
                includes = []
            new_state = {
                "include_sections": includes,
                "show_system_in_preview": bool(show_sys),
            }
            tmpl_state = new_state
            return (
                tmpl_state,
                "✅ Template applied.",
                gr.update(visible=("APIs" in includes)),
                gr.update(visible=("Question" in includes)),
                gr.update(visible=("Code" in includes)),
                gr.update(visible=("Answer" in includes)),
            )

        apply_btn.click(
            _apply_template,
            inputs=[include_sections, show_system_in_preview, template_state],
            outputs=[template_state, template_feedback, apis, question, code, answer],
        )

        # Add example uses current template + system
        add_btn.click(
            add_example_and_summarize_with_template,
            inputs=[dataset_state, template_state, system_global, apis, question, code, answer],
            outputs=[dataset_state, feedback_single, count, dataset_table]
        )

        gr.Markdown(
            "—\n"
            "**Tips**\n"
            "- Type a number or use arrow keys; the index will snap to a valid integer.\n"
            "- Click **View Selected** anytime; the value is clamped and the viewer updates.\n"
            "- Deleting an item keeps the index valid and refreshes the preview."
        )

    return demo

if __name__ == "__main__":
    demo = build_app()
    demo.launch()
