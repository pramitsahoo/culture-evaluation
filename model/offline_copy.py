# import torch
# import transformers
# import time
# from typing import Union, List
# from huggingface_hub import login
# import os

# os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HUGGINGFACE_HUB_TOKEN"



# models = {
#     "llama-3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
#     "llama-2": "meta-llama/Llama-2-7b-chat-hf",
#     "mistral": "mistralai/Mistral-7B-Instruct-v0.3"
# }

# def create_prompt_llama3(user_prompt):
#     # return [
#     #     {"role": "system", "content": "Imagine you are an Indian person. Convert the text to Indian Culture in English. Do not add explanation."},
#     #     {"role": "user", "content": user_prompt},
#     # ]
#     return [
#         {

# "role": "system",
# "content": """
# Adapt the given text to reflect contemporary Indian culture, settings, and context, while preserving the original mathematical problem and numerical values.
# Replace names, locations, currencies, and other cultural references with Indian equivalents.
# Ensure the adapted text maintains clarity, coherence, and cultural sensitivity.
# Specific guidelines:
# Replace names with common Indian names (female for female, male for male)
# Use Indian locations
# Convert currencies to Indian Rupees (INR)
# Remove foreign currency references (USD, etc.)
# Incorporate Indian traditions, festivals, and cultural practices
# Use regional-specific terminology and expressions
# Replace foreign food items with Indian equivalents (e.g., muffins to parathas)
# Maintain original mathematical operations and numerical values
# Provide only the adapted text, without calculations or explanations.
# Include the adapted mathematical problem question.
# """
# },
#         {"role": "user", "content": user_prompt},
#     ]
 

# def create_prompt_llama2(user_prompt):
#     return f"""<s>[INST] <<SYS>>
# Imagine you are an Indian person. Convert the text to Indian Culture in English. Do not add explanation.
# <</SYS>>

# {user_prompt} [/INST]
# """

# def create_prompt_mistral(user_prompt):
#     return f"""<s>[INST] Imagine you are an Indian person. Convert the following text to Indian Culture in English. Do not add explanation:

# {user_prompt} [/INST]
# """

# def get_model_terminators(model_name: str, tokenizer) -> List[int]:
#     common_terminators = [tokenizer.eos_token_id]
    
#     if model_name == "llama-3":
#         return common_terminators + [tokenizer.convert_tokens_to_ids("<|eot_id|>")]
#     elif model_name == "llama-2":
#         return common_terminators + [tokenizer.convert_tokens_to_ids("</s>")]
#     elif model_name == "mistral":
#         return common_terminators + [tokenizer.convert_tokens_to_ids("</s>")]
#     else:
#         return common_terminators

# def call_llm(model_name: str, prompt: Union[str, List[str]], temperature: float, max_new_tokens: int, num_return_sequences: int = 1, stop: Union[str, List[str]] = None):
#     if model_name not in models:
#         raise ValueError(f"Unsupported model: {model_name}")

#     model_id = models[model_name]
#     pipe = transformers.pipeline(
#         "text-generation",
#         model=model_id,
#         model_kwargs={"torch_dtype": torch.bfloat16},
#         device_map="auto",
#         token=os.environ["HUGGING_FACE_HUB_TOKEN"],
#     )

#     # Ensure prompt is a list
#     if isinstance(prompt, str):
#         prompt = [prompt]

#     llm_generations = []
#     start = time.time()

#     for i, prompt_ in enumerate(prompt):
#         if model_name == "llama-3":
#             messages = create_prompt_llama3(prompt_)
#         elif model_name == "llama-2":
#             messages = create_prompt_llama2(prompt_)
#         elif model_name == "mistral":
#             messages = create_prompt_mistral(prompt_)
#         else:
#             raise ValueError(f"Unsupported model: {model_name}")

#         terminators = get_model_terminators(model_name, pipe.tokenizer)

#         try:
#             outputs = pipe(
#                 messages,
#                 max_new_tokens=max_new_tokens,
#                 pad_token_id=pipe.tokenizer.eos_token_id,
#                 eos_token_id=terminators,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_p=0.9,
#                 num_return_sequences=num_return_sequences,
#                 return_full_text=False,  # Only return the newly generated text
#             )

#             for output in outputs:
#                 generated_text = output['generated_text'].strip()
#                 llm_generations.append(generated_text)

#         except Exception as e:
#             print(f"Error processing prompt {i}: {str(e)}")
#             llm_generations.append(None)

#     end = time.time()
#     print("Time elapsed:", end - start)
#     return llm_generations



import torch
import transformers
import time
import re
from typing import Union, List
from huggingface_hub import login
import os

os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HUGGING_FACE_HUB_TOKEN"

models = {
    "llama-3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct"   
}

def create_prompt(model_name: str, user_prompt: str):
    base_instruction = """
Adapt the given text to reflect contemporary Indian culture, settings, and context, while preserving the original mathematical problem and numerical values.
Replace names, locations, currencies, and other cultural references with Indian equivalents.
Ensure the adapted text maintains clarity, coherence, and cultural sensitivity.
Specific guidelines:
- Replace names with common Indian names (female for female, male for male)
- Use Indian locations
- Convert currencies to Indian Rupees (INR)
- Remove foreign currency references (USD, etc.)
- Incorporate Indian traditions, festivals, and cultural practices
- Use regional-specific terminology and expressions
- Replace foreign food items with Indian equivalents (e.g., muffins to parathas)
- Maintain original mathematical operations and numerical values
Provide only the adapted text of the problem statement.
Do not include any calculations, step-by-step solutions, or explanations.
Do not solve the problem or provide the answer.
"""
    if model_name in ["llama-3.2-1b", "llama-3.2-3b"]:
        return f"""<|start_header_id|>system<|end_header_id|>
{base_instruction}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    elif model_name == "llama-3":
        return [
            {"role": "system", "content": base_instruction},
            {"role": "user", "content": user_prompt},
        ]
    elif model_name == "llama-2":
        return f"""<s>[INST] <<SYS>>
{base_instruction}
<</SYS>>

{user_prompt} [/INST]
"""
    elif model_name == "mistral":
        return f"""<s>[INST] {base_instruction}

{user_prompt} [/INST]
"""
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_model_terminators(model_name: str, tokenizer) -> List[int]:
    common_terminators = [tokenizer.eos_token_id]
    if model_name in ["llama-3.2-1b", "llama-3.2-3b"]:
        # Add Llama 3.2 specific terminators
        return common_terminators + [tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    elif model_name == "llama-3":
        return common_terminators + [tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    elif model_name == "llama-2":
        return common_terminators + [tokenizer.convert_tokens_to_ids("</s>")]
    elif model_name == "mistral":
        return common_terminators + [tokenizer.convert_tokens_to_ids("</s>")]
    else:
        return common_terminators

def remove_explanations(text: str) -> str:
    # Remove any text after "Let's calculate" or similar phrases
    explanation_starts = ["Let's calculate", "Let's solve", "To solve this", "The solution is", "Here's how"]
    for start in explanation_starts:
        if start.lower() in text.lower():
            text = text.split(start)[0]
    
    # Remove any step-by-step calculations
    text = re.sub(r'\b\d+\s*[+\-*/]\s*\d+\s*=\s*\d+', '', text)
    
    return text.strip()

def has_explanation(text: str) -> bool:
    explanation_indicators = [
        r'\b(calculation|solve|solution|answer|result)\b',
        r'\b\d+\s*[+\-*/]\s*\d+\s*=\s*\d+',
        r'\b(first|second|third|fourth|fifth|finally)\b.*\bthen\b',
        r'\b(step|stage)\s*\d+',
        r'\btherefore\b'
    ]
    
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in explanation_indicators)

def call_llm(model_name: str, prompt: Union[str, List[str]], temperature: float, max_new_tokens: int, num_return_sequences: int = 1, stop: Union[str, List[str]] = None):
    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}")

    model_id = models[model_name]
    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=os.environ["HUGGING_FACE_HUB_TOKEN"],
    )

    # Ensure prompt is a list
    if isinstance(prompt, str):
        prompt = [prompt]

    llm_generations = []
    start = time.time()

    for i, prompt_ in enumerate(prompt):
        messages = create_prompt(model_name, prompt_)
        terminators = get_model_terminators(model_name, pipe.tokenizer)

        try:
            outputs = pipe(
                messages,
                max_new_tokens=1024,
                pad_token_id=pipe.tokenizer.eos_token_id,
                eos_token_id=terminators,
                do_sample=False,
                temperature=None,
                top_p=None,
                num_return_sequences=num_return_sequences,
                return_full_text=False,  # Only return the newly generated text
            )

            for output in outputs:
                generated_text = output['generated_text'].strip()
                cleaned_text = remove_explanations(generated_text)
                explanation_detected = has_explanation(generated_text)
                llm_generations.append({
                    "adapted_text": cleaned_text,
                    "has_explanation": "Yes" if explanation_detected else "No"
                })

        except Exception as e:
            print(f"Error processing prompt {i}: {str(e)}")
            llm_generations.append({
                "adapted_text": None,
                "has_explanation": "No"
            })

    end = time.time()
    print("Time elapsed:", end - start)
    return llm_generations

