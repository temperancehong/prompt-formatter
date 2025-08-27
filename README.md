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

[Link to current version live demo](https://huggingface.co/spaces/temperancehong/few-shot-prompt-formatter)

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
What modality is this image? The image is in the path path/to/the/image

4. The generated code
<CODE>
def answer_code():
    img_array = read_rgb("path/to/the/image")
    cls = classify_image(img_array)
    return cls
</CODE>

5. Answer: MRI
```

The interface has mainly two parts: 
1. `Single example`: where you enter your text, build your prompt in a click, and then preview the formatted prompt. Also within a click, you can add this single prompt to your few-shot prompt list.
2. `Prompt Set Builder`: you can see the few-shot examples you added to the list, you can export the whole list to a json file.

## Versions
- 28 August 2025: initial version
- To be updated