from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset

torch.cuda.empty_cache()

base_model = "berkeley-nest/Starling-LM-7B-alpha"
dataset_path = "fine_tune_data/arc_aug_train.json"
fine_tuned_model = "finetuned-models/Starling-LM-7B-alpha-finetuned"

dataset = load_dataset("json", data_files=dataset_path, split="train")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    #inference_mode=False,
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config = bnb_config,
    device_map="auto"
)

model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
# model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token
tokenizer.padding_side = "right"

model = get_peft_model(model, peft_config)

steps_per_epoch = len(dataset)//(1*2)

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim ="paged_adamw_8bit",
    save_steps=steps_per_epoch//100,
    save_strategy="steps",
    logging_steps=1,
    learning_rate=2e-4,
    #weight_decay=0.001,
    fp16=True,
    #bf16=False,
    #max_grad_norm=0.3,
    #max_steps=-1,
    #warmup_ratio=0.3,
    warmup_steps=0.03,
    group_by_length=True,
    #lr_scheduler_type="constant"
)

def formatting_prompts_func(example):
    output_texts = []
    temp = "GPT4 Correct User:{prompt_template}\n{prompt}<|end_of_turn|>GPT4 Correct Assistant:{response}<|end_of_turn|>"
    with open("prompt-template.txt", "r") as f:
        prompt_template = f.read()
    
    for i in range(len(example)):
        text = temp.format(prompt_template=prompt_template, prompt=example['prompt'][i], response=example['test_output'][i])
        output_texts.append(text)
    return output_texts

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
    #packing=False,
    formatting_func=formatting_prompts_func
)

#trainer.train()
#trainer.model.save_pretrained(fine_tuned_model)
