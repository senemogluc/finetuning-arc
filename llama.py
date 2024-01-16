from llama_cpp import Llama
import os

#Demo 

llm = Llama(
  model_path="starling-lm-7b-alpha.Q6_K.gguf",
  n_ctx=32768,  
  n_threads_batch=2,
  n_gpu_layers=-1,
  n_batch=1024,
  verbose=False,
)

PROMPT_PATH = "prompts/eval-prompts/"

for prompt in os.listdir(PROMPT_PATH):
    if os.path.exists('outputs/starling-outputs-eval-0.5/' + prompt.replace('prmt', 'comp')):
        print(prompt, "already exists")
        continue

    with open(PROMPT_PATH + prompt) as f:
        prompt_input = f.read()

    if len(prompt_input) > 14000:
        print(prompt, "too long")
        continue

    output = llm(
        f"GPT4 User: {prompt_input}<|end_of_turn|>GPT4 Assistant:", 
        max_tokens=2048, 
        stop=["</s>"],  
        temperature=0.5,
    )
        
    with open('outputs/starling-outputs-eval-0.5/' + prompt.replace('prmt', 'comp'), mode='w+') as f:
        f.write(output["choices"][0]['text'].strip())
     
    print(prompt + " done")

# with open('prompts/concept-prompts/TopBottom2D2_prmt.txt') as f:
#     prompt = f.read()

# # Simple inference example
# output = llm(
#   f"GPT4 User: {prompt}<|end_of_turn|>GPT4 Assistant:", # Prompt
#   max_tokens=2048,  # Generate up to 2048 tokens
#   stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
#   temperature=0.5,
# )

# print(output["choices"][0]['text'].strip())



"""
for prompt in os.listdir('data/training/train-prompts'):
    print(prompt) #a64e4611_prmt.txt
    with open('data/training/train-prompts/' + prompt) as f:
        prompt_input = f.read()
        print(prompt_input) #True
        #time.sleep(10)

    
    output = llm(
            f"GPT4 User: {prompt_input}<|end_of_turn|>GPT4 Assistant:", # Prompt
            max_tokens=2048,  # Generate up to 2048 tokens
            stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
            temperature=0.5
            )
    
    print(output["choices"][0]['text'].strip())
"""

# Chat Completion API
"""
llm = Llama(model_path="starling-lm-7b-alpha.Q6_K.gguf", chat_format="llama-2")  # Set chat_format according to the model you are using
llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are a story writing assistant."},
        {
            "role": "user",
            "content": "Write a story about llamas."
        }
    ]
)
"""

