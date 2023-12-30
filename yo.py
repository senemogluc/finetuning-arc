from datasets import load_dataset

dataset = load_dataset("json", data_files = "fine_tune_data/arc_aug_train.json", split="train")

print(dataset['prompt'][0], dataset['test_output'][0])

# print(dataset['train']['prompt'][:5])