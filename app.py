# app.py
import gradio as gr
from builder_utils import (
    DEFAULT_SYSTEM,
    DEFAULT_APIS,
    DEFAULT_QUESTION,
    DEFAULT_IMAGE_PATH,
    DEFAULT_CODE,
    DEFAULT_ANSWER,
    COLUMNS,
    render_preview,
    add_example_and_summarize,
    get_example_detail,
    delete_example_and_summarize,  # NEW
    export_jsonl,
)

def build_app():
    with gr.Blocks(title="Few-shot Prompt Builder (Medical VQA)", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## 🧪 Few-shot Prompt Builder for Medical VQA\n"
            "Compose consistent few-shot examples (SYSTEM, Tool APIs, Question, Code, Answer), preview them, "
            "add to a dataset, visualize it, and export JSONL."
        )

        dataset_state = gr.State([])  # list of JSON records

        with gr.Tabs():
            # ---- Tab 1: Single Example ----
            with gr.TabItem("Single Example"):
                with gr.Row():
                    with gr.Column(scale=1):
                        system = gr.Textbox(
                            label="1) Instruction (SYSTEM)",
                            value=DEFAULT_SYSTEM,
                            lines=6,
                            placeholder="Paste your SYSTEM prompt here..."
                        )
                        apis = gr.Code(
                            label="2) Tool APIs (Python doc / signatures)",
                            value=DEFAULT_APIS,
                            language="python",
                            lines=16
                        )
                        question = gr.Textbox(
                            label="3) VQA Question",
                            value=DEFAULT_QUESTION,
                            lines=2
                        )
                        image_path = gr.Textbox(
                            label="(Optional) Image path to inject into the question",
                            value=DEFAULT_IMAGE_PATH,
                            lines=1
                        )
                        code = gr.Code(
                            label="4) Generated Code (the callable plan)",
                            value=DEFAULT_CODE,
                            language="python",
                            lines=10
                        )
                        answer = gr.Textbox(
                            label="5) Answer",
                            value=DEFAULT_ANSWER,
                            lines=1
                        )

                        with gr.Row():
                            preview_btn = gr.Button("🔎 Preview Example", variant="secondary")
                            add_btn = gr.Button("➕ Add Current Example to Dataset", variant="primary")

                        feedback_single = gr.Textbox(label="Status", lines=2, interactive=False)

                    with gr.Column(scale=1):
                        preview = gr.Textbox(
                            label="Formatted Example Preview",
                            lines=32
                        )
                        copy_helper = gr.Textbox(
                            label="(Hidden) Raw text to copy",
                            visible=False
                        )
                        gr.Markdown("Tip: After previewing, click the copy icon on the preview box to copy the text.")

                # Preview wiring
                preview_btn.click(
                    render_preview,
                    inputs=[system, apis, question, image_path, code, answer],
                    outputs=[preview, copy_helper]
                )

            # ---- Tab 2: Dataset Builder ----
            with gr.TabItem("Prompt Set Builder"):
                with gr.Row():
                    with gr.Column(scale=1):
                        count = gr.Number(label="Examples in dataset", value=0, precision=0, interactive=False)
                        dataset_table = gr.Dataframe(
                            headers=COLUMNS,
                            value=[],
                            wrap=True,
                            interactive=False,
                            label="Dataset (truncated previews)"
                        )
                        with gr.Row():
                            view_index = gr.Number(
                                label="Select example # (1-based)",
                                value=1,
                                precision=0
                            )
                            view_btn = gr.Button("👁️ View Selected")
                            delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")  # NEW
                        builder_feedback = gr.Textbox(label="Dataset Status", lines=2, interactive=False)  # NEW
                    with gr.Column(scale=1):
                        full_view = gr.Textbox(
                            label="Selected Example (full)",
                            lines=34
                        )

                # View wiring
                view_btn.click(
                    get_example_detail,
                    inputs=[dataset_state, view_index],
                    outputs=[full_view]
                )

                # Delete wiring (updates state, status, count, and table)
                delete_btn.click(
                    delete_example_and_summarize,
                    inputs=[dataset_state, view_index],
                    outputs=[dataset_state, builder_feedback, count, dataset_table]
                )

                with gr.Row():
                    export_btn = gr.Button("📤 Export JSONL")
                    export_file = gr.File(label="Download JSONL", interactive=False)
                    export_status = gr.Textbox(label="Export Status", lines=2)
                export_btn.click(
                    export_jsonl,
                    inputs=[dataset_state],
                    outputs=[export_file, export_status]
                )

        # IMPORTANT: wire the "Add" button AFTER dataset_table is defined so we can update it
        add_btn.click(
            add_example_and_summarize,
            inputs=[dataset_state, system, apis, question, image_path, code, answer],
            outputs=[dataset_state, feedback_single, count, dataset_table]
        )

        gr.Markdown(
            "—\n"
            "**Notes**\n"
            "- Add examples from the **Single Example** tab.\n"
            "- Use the index field to **view** or **delete** a specific example.\n"
            "- The dataset table shows truncated previews; use the right pane to see the full record."
        )
    return demo

if __name__ == "__main__":
    demo = build_app()
    demo.launch()
