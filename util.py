import json
import os

def task_to_prompt(task_json):
    with open(task_json, mode='r') as f:
        parsed_json = json.load(f)

    inputs = []
    outputs = []

    for case in parsed_json['train']:
        inputs.append("[" + ", ".join([f"{str(row)}" for row in case['input']]) + "]")
        outputs.append("[" + ", ".join([f"{str(row)}" for row in case['output']]) + "]")

    test_input = "[" + ", ".join([f"{str(row)}" for row in parsed_json['test'][0]['input']]) + "]"
    test_output = "[" + ", ".join([f"{str(row)}" for row in parsed_json['test'][0]['output']]) + "]"

    prompt = ""

    for i in range(len(inputs)):
        prompt += "Case " + str(i) + ":\nInput:\n" + inputs[i]
        prompt += "\n\nOutput:\n" + outputs[i] + "\n\n"
    
    prompt += "Test Case:\nInput:\n" + test_input
    prompt += "\n\nOutput:"

    return prompt, test_output



def multiple_tasks_to_prompt(tasks_dir, out_json_dir):
    json_dict = []
    for task in os.listdir(tasks_dir):
        prompt, test_output = task_to_prompt(os.path.join(tasks_dir, task))
        json_dict.append({"prompt": prompt, "test_output": test_output})
    
    with open(out_json_dir, mode='w') as f:
        if not os.path.exists(os.path.dirname(out_json_dir)):
            os.mkdir(os.path.dirname(out_json_dir))

        json.dump(json_dict, f)


            

if __name__ == "__main__":
    # print(task_to_prompt("data/training/0a938d79.json")[0])
    multiple_tasks_to_prompt("data/evaluation", "fine_tune_data/arc_aug_eval.json")