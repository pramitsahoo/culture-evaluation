import torch
import transformers
import time
from typing import Union, List

models = {
    "llama-3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
}

def create_prompt_llama3(user_prompt):
    # return [
    #     {"role": "system", "content": "Imagine you are an Indian person. Convert the text to Indian Culture in English. Do not add explanation."},
    #     {"role": "user", "content": user_prompt},
    # ]
    return [
        # {"role": "system", "content":"Provide only the final numerical answer, without any calculations, explanations, or intermediate steps."
# },           
        {"role": "system", "content":"LLaMA 3.1 8B, FINAL ANSWER ONLY. No calculations, text, units, symbols, or currency signs. JUST THE NUMBER."
        },
        {"role": "user", "content": user_prompt},
    ]

def create_prompt_llama2(user_prompt):
    return f"""<s>[INST] <<SYS>>
Imagine you are an Indian person. Convert the text to Indian Culture in English. Do not add explanation.
<</SYS>>

{user_prompt} [/INST]
"""

def create_prompt_mistral(user_prompt):
    return f"""<s>[INST] Imagine you are an Indian person. Convert the following text to Indian Culture in English. Do not add explanation:

{user_prompt} [/INST]
"""

def get_model_terminators(model_name: str, tokenizer) -> List[int]:
    common_terminators = [tokenizer.eos_token_id]
    
    if model_name == "llama-3":
        return common_terminators + [tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    elif model_name == "llama-2":
        return common_terminators + [tokenizer.convert_tokens_to_ids("</s>")]
    elif model_name == "mistral":
        return common_terminators + [tokenizer.convert_tokens_to_ids("</s>")]
    else:
        return common_terminators

def call_llm(model_name: str, prompt: Union[str, List[str]], temperature: float, max_new_tokens: int, num_return_sequences: int = 1, stop: Union[str, List[str]] = None):
    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}")

    model_id = models[model_name]
    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    # Ensure prompt is a list
    if isinstance(prompt, str):
        prompt = [prompt]

    llm_generations = []
    start = time.time()

    for i, prompt_ in enumerate(prompt):
        if model_name == "llama-3":
            messages = create_prompt_llama3(prompt_)
        elif model_name == "llama-2":
            messages = create_prompt_llama2(prompt_)
        elif model_name == "mistral":
            messages = create_prompt_mistral(prompt_)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        terminators = get_model_terminators(model_name, pipe.tokenizer)

        try:
            outputs = pipe(
                messages,
                max_new_tokens=max_new_tokens,
                pad_token_id=pipe.tokenizer.eos_token_id,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=num_return_sequences,
                return_full_text=False,  # Only return the newly generated text
            )

            for output in outputs:
                generated_text = output['generated_text'].strip()
                llm_generations.append(generated_text)

        except Exception as e:
            print(f"Error processing prompt {i}: {str(e)}")
            llm_generations.append(None)

    end = time.time()
    print("Time elapsed:", end - start)
    return llm_generations

