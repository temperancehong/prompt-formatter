---
title: "VQA Few-shot Prompt Builder"
emoji: "🩺"
colorFrom: "indigo"
colorTo: "blue"
sdk: "gradio"
python_version: "3.10"
sdk_version: "5.43.1"
suggested_hardware: "cpu-basic"
suggested_storage: "small"
app_file: "app.py"
fullWidth: true
header: "mini"
short_description: "Faster few-shot formatting for VQA."
tags:
  - gradio
  - prompt-engineering
  - vqa
  - medical-imaging
  - few-shot-learning
pinned: false
license: "mit"
disable_embedding: false
---


# Few-shot Prompt Builder

This is a helpful tool that I built to facilitate Python code synthesis VQA tasks using few-shot prompting techniques. It allows you to format prompts.

[Link to current version huggingface live demo](https://huggingface.co/spaces/temperancehong/few-shot-prompt-formatter)

For each example, it contains `apis`, `question`, `code` and `answer` as key choices. You can select a few of them or all of them for each example.

```json
{
  "system": "…",
  "example": [
    {"apis": "…", "question": "…", "code": "…", "answer": "…"},
    {"question": "…", "answer": "…"}
  ]
}
```

For now the prompt is in the following format:
```PlainText
1. Instruction
<SYSTEM>You are a helpful radiology assistant. Here I have one image, and please answer me the question by first providing your step by step thoughts in natural language using the supported actions provided in the Python code APIs, and then output the corresponding consistant code that use the APIs provided step by step. Finally, you execute the code, show me the intermediate results and the final answer, and then give me the response.</SYSTEM>

2. Tool APIs
<APIs>
# something in style API doc in python
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
</APIs>

3. The VQA question:
What modality is this image?

4. The generated code
<CODE>
def answer_code():
    cls = classify_image(img_array)
    return cls
</CODE>

5. Answer: MRI
```

The interface has mainly two parts: 
1. `Template`: where you choose your template for building the examples. You can then click `✅ Apply template settings` to apply and continue to the next Tab.
1. `Single example`: where you enter your text, build your prompt in a click, and then preview the formatted prompt. Also within a click, you can add this single prompt to your few-shot prompt list.
2. `Prompt Set Builder`: you can see the few-shot examples you added to the list, you can export the whole list to a `JSON` file. You can also export to `JSONL` files, where each line is one example, where each lines has the system prompts.

## Versions
- 28 August 2025: initial version
- 29 August 2025: With the display prompt example, you can select which fields to fill in the example