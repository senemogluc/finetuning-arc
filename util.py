import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data/training"
PROMPTS_DIR = DATA_DIR / "train-prompts"
OUTPUTS_DIR = DATA_DIR / "train-outputs"


def task_to_prompt(task_json):
    with open(DATA_DIR / task_json) as user_file:
        parsed_json = json.load(user_file)

    inputs = []
    outputs = []

    for case in parsed_json['train']:
        inputs.append("\n".join([f"  {str(row)}" for row in case['input']]))
        outputs.append("\n".join([f"  {str(row)}" for row in case['output']]))

    test_input = "\n".join([f"  {str(row)}" for row in parsed_json['test'][0]['input']])
    test_output = "\n".join([f"  {str(row)}" for row in parsed_json['test'][0]['output']])

    with open('prompt-template.txt', mode='r') as f:
        template = f.read()

    name, ext = os.path.splitext(task_json)

    with open(PROMPTS_DIR / f"{name}_prmt.txt", mode='w+') as f:
        f.write(template)
        for i in range(len(inputs)):
            f.write("Case " + str(i) + ":\nInput:\n" + inputs[i])
            f.write("\n\nOutput:\n" + outputs[i] + "\n\n")
        
        f.write("Case " + str(i+1) + "\nInput:\n" + test_input)
        f.write("\n\nOutput:\n")
        f.write("\nWhat is the output of the last input?")

    with open(OUTPUTS_DIR / f"{name}_out.txt", mode='w+') as f:
        f.write(test_output)


for task in os.listdir(DATA_DIR):
    if task.endswith(".json"):
        task_to_prompt(task)

        #fillednotfilledminimal, topbottom2d2, extractobjectsminimal, movetoboundaryminimal, order10 - samedifferent10-