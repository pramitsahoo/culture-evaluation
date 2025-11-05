# import torch
# import transformers
# import time
# import re
# from typing import Union, List
# from huggingface_hub import login
# import os

# os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HUGGINGFACE_HUB_TOKEN"

# models = {
#     "llama-3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
#     "llama-2": "meta-llama/Llama-2-7b-chat-hf",
#     "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
#     "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
#     "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct"   
# }

# def create_prompt(model_name: str, user_prompt: str):
#     base_instruction = """
# Adapt the given text to reflect contemporary Indian culture, settings, and context, while preserving the original mathematical problem and numerical values.
# Replace names, locations, currencies, and other cultural references with Indian equivalents.
# Ensure the adapted text maintains clarity, coherence, and cultural sensitivity.
# Specific guidelines:
# - Replace names with common Indian names (female for female, male for male)
# - Use Indian locations
# - Convert currencies to Indian Rupees (INR)
# - Remove foreign currency references (USD, etc.)
# - Incorporate Indian traditions, festivals, and cultural practices
# - Use regional-specific terminology and expressions
# - Replace foreign food items with Indian equivalents (e.g., muffins to parathas)
# - Maintain original mathematical operations and numerical values
# Provide only the adapted text of the problem statement.
# Provide the replaced words on a separate column naming replaced words.
# Do not include any calculations, step-by-step solutions, or explanations.
# Do not solve the problem or provide the answer.
# """
#     if model_name in ["llama-3.2-1b", "llama-3.2-3b"]:
#         return f"""<|start_header_id|>system<|end_header_id|>
# {base_instruction}
# <|eot_id|><|start_header_id|>user<|end_header_id|>
# {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
#     elif model_name == "llama-3":
#         return [
#             {"role": "system", "content": base_instruction},
#             {"role": "user", "content": user_prompt},
#         ]
#     elif model_name == "llama-2":
#         return f"""<s>[INST] <<SYS>>
# {base_instruction}
# <</SYS>>

# {user_prompt} [/INST]
# """
#     elif model_name == "mistral":
#         return f"""<s>[INST] {base_instruction}

# {user_prompt} [/INST]
# """
#     else:
#         raise ValueError(f"Unsupported model: {model_name}")

# def get_model_terminators(model_name: str, tokenizer) -> List[int]:
#     common_terminators = [tokenizer.eos_token_id]
#     if model_name in ["llama-3.2-1b", "llama-3.2-3b"]:
#         # Add Llama 3.2 specific terminators
#         return common_terminators + [tokenizer.convert_tokens_to_ids("<|eot_id|>")]
#     elif model_name == "llama-3":
#         return common_terminators + [tokenizer.convert_tokens_to_ids("<|eot_id|>")]
#     elif model_name == "llama-2":
#         return common_terminators + [tokenizer.convert_tokens_to_ids("</s>")]
#     elif model_name == "mistral":
#         return common_terminators + [tokenizer.convert_tokens_to_ids("</s>")]
#     else:
#         return common_terminators

# def remove_explanations(text: str) -> str:
#     # Remove any text after "Let's calculate" or similar phrases
#     explanation_starts = ["Let's calculate", "Let's solve", "To solve this", "The solution is", "Here's how"]
#     for start in explanation_starts:
#         if start.lower() in text.lower():
#             text = text.split(start)[0]
    
#     # Remove any step-by-step calculations
#     text = re.sub(r'\b\d+\s*[+\-*/]\s*\d+\s*=\s*\d+', '', text)
    
#     return text.strip()

# def has_explanation(text: str) -> bool:
#     explanation_indicators = [
#         r'\b(calculation|solve|solution|answer|result)\b',
#         r'\b\d+\s*[+\-*/]\s*\d+\s*=\s*\d+',
#         r'\b(first|second|third|fourth|fifth|finally)\b.*\bthen\b',
#         r'\b(step|stage)\s*\d+',
#         r'\btherefore\b'
#     ]
    
#     return any(re.search(pattern, text, re.IGNORECASE) for pattern in explanation_indicators)

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
#         messages = create_prompt(model_name, prompt_)
#         terminators = get_model_terminators(model_name, pipe.tokenizer)

#         try:
#             outputs = pipe(
#                 messages,
#                 max_new_tokens=1024,
#                 pad_token_id=pipe.tokenizer.eos_token_id,
#                 eos_token_id=terminators,
#                 do_sample=False,
#                 temperature=None,
#                 top_p=None,
#                 num_return_sequences=num_return_sequences,
#                 return_full_text=False,  # Only return the newly generated text
#             )

#             for output in outputs:
#                 generated_text = output['generated_text'].strip()
#                 cleaned_text = remove_explanations(generated_text)
#                 explanation_detected = has_explanation(generated_text)
#                 llm_generations.append({
#                     "adapted_text": cleaned_text,
#                     "has_explanation": "Yes" if explanation_detected else "No"
#                 })

#         except Exception as e:
#             print(f"Error processing prompt {i}: {str(e)}")
#             llm_generations.append({
#                 "adapted_text": None,
#                 "has_explanation": "No"
#             })

#     end = time.time()
#     print("Time elapsed:", end - start)
#     return llm_generations


##############################################################################################################################
#                   This Script is for generating the replaced word along with the adapted text                             #
##############################################################################################################################

# import torch
# import transformers
# import time
# from typing import Union, List
# from huggingface_hub import login
# import os

# # Set your Hugging Face Hub token
# os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HUGGINGFACE_HUB_TOKEN"

# # Define the models with their corresponding Hugging Face identifiers
# models = {
#     "llama-3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
#     "llama-2": "meta-llama/Llama-2-7b-chat-hf",
#     "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
#     "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
#     "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
#     "llama-3-nanda": "MBZUAI/Llama-3-Nanda-10B-Chat"  
# }
# # Provide only the replaced words on a separate column naming replaced words in a valid json format.
# # Here is an example on how you should give me the two things i asked as output -
# def create_prompt(model_name: str, user_prompt: str):
#     base_instruction = """
# Adapt the given text to reflect contemporary Indian culture, settings, and context, while preserving the original mathematical problem and numerical values.
# Replace names, locations, currencies, and other cultural references with Indian equivalents.
# Ensure the adapted text maintains clarity, coherence, and cultural sensitivity.
# Specific guidelines:
# - Replace names with common Indian names (female for female, male for male)
# - Use Indian locations
# - Convert currencies to Indian Rupees (INR)
# - Remove foreign currency references (USD, etc.)
# - Incorporate Indian traditions, festivals, and cultural practices
# - Use regional-specific terminology and expressions
# - Replace foreign food items with Indian equivalents (e.g., muffins to parathas)
# - Maintain original mathematical operations and numerical values
# Provide only the adapted text and replaced words of the problem statement in a valid json format just like this example - 
# [
#     {
#         "adapted_text": "Ramesh bought 5 parathas for ₹830 and gave 2 to his friend. How much money did he spend per paratha?",
#         "replaced_words": {"John": "Ramesh", "muffins": "parathas", "$10": "₹830"}
#     }
# ]

# Do not include any calculations, step-by-step solutions, or explanations.
# Do not solve the problem or provide the answer.
# Do not hallucinate.
# """
#     # Print the prompt for debugging
#     print(f"\n--- Prompt Creation for {model_name} ---")
    
#     if model_name in ["llama-3.2-1b", "llama-3.2-3b"]:
#         prompt = f"""<|start_header_id|>system<|end_header_id|>
# {base_instruction}
# <|eot_id|><|start_header_id|>user<|end_header_id|>
# {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
#         print(prompt)
#         return prompt
#     elif model_name == "llama-3-nanda":
#         # Specific prompt format for Llama-3-Nanda
#         prompt_hindi = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>{Question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
#         return prompt_hindi.format_map({'Question': user_prompt})
#     elif model_name == "llama-3":
#         return [
#             {"role": "system", "content": base_instruction},
#             {"role": "user", "content": user_prompt},
#         ]
#     elif model_name == "llama-2":
#         return f"""<s>[INST] <<SYS>>
# {base_instruction}
# <</SYS>>

# {user_prompt} [/INST]
# """
#     elif model_name == "mistral":
#         return f"""<s>[INST] {base_instruction}

# {user_prompt} [/INST]
# """
#     else:
#         raise ValueError(f"Unsupported model: {model_name}")

# def get_model_terminators(model_name: str, tokenizer) -> List[int]:
#     common_terminators = [tokenizer.eos_token_id]
#     if model_name in ["llama-3.2-1b", "llama-3.2-3b", "llama-3-nanda"]:
#         # Add Llama 3.2 specific terminators
#         return common_terminators + [tokenizer.convert_tokens_to_ids("<|eot_id|>")]
#     elif model_name == "llama-3":
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
    
#     # Define additional arguments for specific models
#     model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    
#     if model_name == "llama-3-nanda":
#         model_kwargs["trust_remote_code"] = True  # Required for custom model classes


#     # model_id = models[model_name]
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
#         messages = create_prompt(model_name, prompt_)
#         terminators = get_model_terminators(model_name, pipe.tokenizer)

#         try:
#             outputs = pipe(
#                 messages,
#                 max_new_tokens=max_new_tokens,
#                 pad_token_id=pipe.tokenizer.eos_token_id,
#                 eos_token_id=terminators,
#                 do_sample=False,
#                 temperature=None,
#                 top_p=None,  # Typically used for nucleus sampling
#                 num_return_sequences=num_return_sequences,
#                 return_full_text=False,  # Only return the newly generated text
#             )

#             for output in outputs:
#                 generated_text = output['generated_text'].strip()
#                 # llm_generations.append({
#                 #     "adapted_text": generated_text
#                 # })
#                 llm_generations.append(generated_text)

#         except Exception as e:
#             print(f"Error processing prompt {i}: {str(e)}")
#             # llm_generations.append({
#             #     "adapted_text": None
#             # })
#             llm_generations.append(generated_text)

#     end = time.time()
#     print("Time elapsed:", end - start)
#     return llm_generations


# ######################################################################################
# import torch
# import transformers
# import time
# from typing import Union, List
# from huggingface_hub import login
# import os

# # Set your Hugging Face Hub token
# os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HUGGINGFACE_HUB_TOKEN"

# # Define the models with their corresponding Hugging Face identifiers
# MODELS = {
#     "llama-3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
#     "llama-2": "meta-llama/Llama-2-7b-chat-hf",
#     "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
#     "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
#     "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
#     "llama-3-nanda": "MBZUAI/Llama-3-Nanda-10B-Chat",
#     "gemma-2-9b": "google/gemma-2-9b-it",
#     "gemma-2-2b": "google/gemma-2-2b-it",
# }



# def get_system_prompt():
#     """Returns the system prompt for cultural adaptation."""
#     return """You are an AI assistant tasked with adapting a text to suit a specific cultural context in a particular language while maintaining the original meaning and intent. Your goal is to ensure the text feels natural and appropriate for the target audience, considering cultural nuances, values, and sensitivities. When making these adaptations, follow these key steps:

# 1. Cultural Relevance: Adjust any idioms, metaphors, or cultural references that may not resonate with the target audience. Replace them with culturally appropriate alternatives.
# 2. Tone and Intent: Preserve the original emotional tone and message, even when making cultural adjustments.
# 3. Cultural Sensitivity: Be mindful of topics, words, or phrases that might be sensitive or inappropriate in the target culture.

# Specific guidelines:
# - Replace foreign names with common Indian names (female for female, male for male).Use a diverse set of names.
# - Use Indian locations in place of foreign locations.
# - Convert all foreign currencies to Indian Rupees (INR) using a clear symbolic or approximate rate (for example, $1 = ₹83). Remove any remaining references to foreign currency (USD, etc.).
# - Incorporate Indian traditions, festivals, and cultural practices only if it is contextually appropriate and does not distort the original meaning.
# - Use regional-specific terminology and expressions without changing the logical sense of the text.
# - Replace foreign food items with Indian equivalents only if it makes sense. For example, “muffins” can become “parathas,” but do not replace food items that are already commonly used in India or are essential to the text’s logic (e.g., do not replace “eggs” if it’s about chickens laying eggs).
# - Maintain original mathematical operations and numerical values. Do not show any calculations or provide step-by-step solutions.
# - Do not transliterate.
# - Do not solve the problem or provide the answer.
# - Do not hallucinate or introduce factual errors.
# - Ensure the adapted text is coherent and flows naturally in its new cultural context.
# - Provide your response as a single-line JSON array without any line breaks or extra whitespace. The response must be valid JSON that can be parsed directly.
# - For the replaced_concepts dictionary:
#     - ONLY include terms that were actually changed to different terms (e.g., "John":"Ramesh").
#     - DO NOT include terms that remained the same (e.g., do not include "eggs":"eggs" if it was not changed).
#     - Use the ₹ symbol directly in the values, not Unicode escape sequences.
#     - Include ONLY the meaningful substitutions you made (e.g., {"John":"Ramesh","muffins":"parathas","$10":"₹830"}).

# Example of correct replaced_concepts:
# Good: {"John":"Ramesh", "muffins":"parathas", "$10":"₹830"}
# Bad: {"eggs":"eggs", "John":"Ramesh", "muffins":"parathas", "$10":"\\u20b9830"}

# Provide only the adapted text and replaced words of the problem statement in a valid JSON format, like this example:
# [{"cultural_adapted_text":"Ramesh bought 5 parathas for ₹830 and gave 2 to his friend. How much money did he spend per paratha?","replaced_concepts":{"John":"Ramesh","muffins":"parathas","$10":"₹830"}}]"""




# def create_prompt(model_name: str, user_prompt: str) -> Union[str, List[dict]]:
#     """Creates model-specific prompts with the cultural adaptation instruction."""
#     system_prompt = get_system_prompt()
#     # user_message = f"Adapt the following text to Indian culture and respond ONLY with the JSON array:\n{user_prompt}"
#     user_message = f"Adapt the following text to Indian culture and provide the output in the specified single-line format:\n{user_prompt}"
#     if model_name in ["gemma-2-9b", "gemma-2-2b"]:
#         return [{"role": "user", "content": f"{system_prompt}\n\n{user_message}"}]
    
#     elif model_name in ["llama-3.2-1b", "llama-3.2-3b"]:
#         return f"""<|start_header_id|>system<|end_header_id|>
# {system_prompt}
# <|eot_id|><|start_header_id|>user<|end_header_id|>
# {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
#     elif model_name == "llama-3-nanda":
#         return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
#     elif model_name == "llama-3":
#         return [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_message},
#         ]
    
#     elif model_name == "llama-2":
#         return f"""<s>[INST] <<SYS>>
# {system_prompt}
# <</SYS>>

# {user_message} [/INST]"""
    
#     elif model_name == "mistral":
#         return f"""<s>[INST] {system_prompt}

# {user_message} [/INST]"""
    
#     else:
#         raise ValueError(f"Unsupported model: {model_name}")

# def get_model_terminators(model_name: str, tokenizer) -> List[int]:
#     """Gets model-specific termination tokens."""
#     common_terminators = [tokenizer.eos_token_id]
    
#     terminator_mapping = {
#         "llama-3.2-1b": "<|eot_id|>",
#         "llama-3.2-3b": "<|eot_id|>",
#         "llama-3-nanda": "<|eot_id|>",
#         "llama-3": "<|eot_id|>",
#         "llama-2": "</s>",
#         "mistral": "</s>",
#         "gemma-2-9b": "<end_of_turn>" ,
#         "gemma-2-2b": "<end_of_turn>",
#     }
    
#     if model_name in terminator_mapping:
#         return common_terminators + [tokenizer.convert_tokens_to_ids(terminator_mapping[model_name])]
#     return common_terminators

# def call_llm(
#     model_name: str,
#     prompt: Union[str, List[str]],
#     temperature: float = None,
#     max_new_tokens: int = 2048,
#     num_return_sequences: int = 1
# ) -> List[str]:
#     """
#     Calls the specified language model to perform cultural adaptation.
    
#     Args:
#         model_name: Name of the model to use
#         prompt: Input prompt or list of prompts
#         temperature: Sampling temperature (0.0 to 1.0)
#         max_new_tokens: Maximum number of tokens to generate
#         num_return_sequences: Number of alternative generations to return
    
#     Returns:
#         List of generated responses
#     """
#     if model_name not in MODELS:
#         raise ValueError(f"Unsupported model: {model_name}")

#     model_id = MODELS[model_name]
#     model_kwargs = {
#         "torch_dtype": torch.bfloat16,
#         "device_map": "auto"
#     }
    
#     if model_name == "llama-3-nanda":
#         model_kwargs["trust_remote_code"] = True

#     # Initialize the pipeline without device_map in pipeline args
#     pipe = transformers.pipeline(
#         "text-generation",
#         model=model_id,
#         model_kwargs=model_kwargs,
#         token=os.environ["HUGGING_FACE_HUB_TOKEN"],
#     )

#     # Ensure prompt is a list
#     if isinstance(prompt, str):
#         prompt = [prompt]

#     generations = []
#     start_time = time.time()

#     for i, single_prompt in enumerate(prompt):
#         messages = create_prompt(model_name, single_prompt)
#         terminators = get_model_terminators(model_name, pipe.tokenizer)

#         try:
#             outputs = pipe(
#                 messages,
#                 max_new_tokens=max_new_tokens,
#                 pad_token_id=pipe.tokenizer.eos_token_id,
#                 eos_token_id=terminators,
#                 do_sample=False,
#                 temperature=None,
#                 top_p=None,
#                 num_return_sequences=num_return_sequences,
#                 return_full_text=False,
#             )

#             for output in outputs:
#                 generated_text = output['generated_text'].strip()
#                 generations.append(generated_text)

#         except Exception as e:
#             print(f"Error processing prompt {i}: {str(e)}")
#             generations.append(None)

#     elapsed_time = time.time() - start_time
#     print(f"Generation completed in {elapsed_time:.2f} seconds")
    
#     return generations



################################################## With deepseek ####################################
import torch
import transformers
import time
from typing import Union, List
from huggingface_hub import login
import os

# Set your Hugging Face Hub token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HUGGING_FACE_HUB_TOKEN"

# Define the models with their corresponding Hugging Face identifiers,
# including the new DeepSeek-R1-Distill models.
MODELS = {
    "llama-3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3-nanda": "MBZUAI/Llama-3-Nanda-10B-Chat",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma-2-2b": "google/gemma-2-2b-it",
    "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}

def is_deepseek_model(model_name: str) -> bool:
    """Checks if the model is from the DeepSeek family."""
    return model_name.startswith("deepseek")

def get_system_prompt():
    """Returns the system prompt for cultural adaptation with strict JSON-only output instructions."""
    base_prompt = """You are an AI assistant tasked with adapting a text to suit a specific cultural context in a particular language while maintaining the original meaning and intent. Your goal is to ensure the text feels natural and appropriate for the target audience, considering cultural nuances, values, and sensitivities. When making these adaptations, follow these key steps:

1. Cultural Relevance: Adjust any idioms, metaphors, or cultural references that may not resonate with the target audience. Replace them with culturally appropriate alternatives.
2. Tone and Intent: Preserve the original emotional tone and message, even when making cultural adjustments.
3. Cultural Sensitivity: Be mindful of topics, words, or phrases that might be sensitive or inappropriate in the target culture.

Specific guidelines:
- Replace foreign names with common Indian names (female for female, male for male). Use a diverse set of names.
- Use Indian locations in place of foreign locations.
- Convert all foreign currencies to Indian Rupees (INR) using a clear symbolic or approximate rate (for example, $1 = ₹83). Remove any remaining references to foreign currency (USD, etc.).
- Incorporate Indian traditions, festivals, and cultural practices only if it is contextually appropriate and does not distort the original meaning.
- Use regional-specific terminology and expressions without changing the logical sense of the text.
- Replace foreign food items with Indian equivalents only if it makes sense. For example, “muffins” can become “parathas,” but do not replace food items that are already commonly used in India or are essential to the text’s logic.
- Maintain original mathematical operations and numerical values. Do not show any calculations or provide step-by-step solutions.
- Do not transliterate.
- Do not solve the problem or provide the answer.
- Do not hallucinate or introduce factual errors.
- Ensure the adapted text is coherent and flows naturally in its new cultural context.
- Provide your response as a single-line JSON array without any line breaks or extra whitespace. The response must be valid JSON that can be parsed directly.
- For the replaced_concepts dictionary:
    - ONLY include terms that were actually changed to different terms (e.g., "John":"Ramesh").
    - DO NOT include terms that remained the same (e.g., do not include "eggs":"eggs" if it was not changed).
    - Use the ₹ symbol directly in the values, not Unicode escape sequences.
    - Include ONLY the meaningful substitutions you made (e.g., {"John":"Ramesh","muffins":"parathas","$10":"₹830"}).

Important: You must output only a single-line JSON array wrapped in <output> and </output> tags, and nothing else. Do not include any internal chain-of-thought or explanations.

Example of correct replaced_concepts:
Good: {"John":"Ramesh", "muffins":"parathas", "$10":"₹830"}
Bad: {"eggs":"eggs", "John":"Ramesh", "muffins":"parathas", "$10":"\\u20b9830"}

Provide only the adapted text and replaced words of the problem statement in a valid JSON format, like this example:
<output>[{"cultural_adapted_text":"Ramesh bought 5 parathas for ₹830 and gave 2 to his friend. How much money did he spend per paratha?","replaced_concepts":{"John":"Ramesh","muffins":"parathas","$10":"₹830"}}]</output>
"""
    return base_prompt

def create_prompt(model_name: str, user_prompt: str) -> Union[str, List[dict]]:
    """Creates model-specific prompts with the cultural adaptation instruction."""
    system_prompt = get_system_prompt()
    # For DeepSeek models, add an explicit instruction to hide internal chain-of-thought details.
    if is_deepseek_model(model_name):
        system_prompt += "\n\nDO NOT include any internal chain-of-thought or reasoning details in your final output. Only output the final JSON answer as specified wrapped in <output> tags."
    
    user_message = f"Adapt the following text to Indian culture and provide the output in the specified format:\n{user_prompt}"
    
    # Branching for specific models based on expected formatting...
    if model_name in ["gemma-2-9b", "gemma-2-2b"]:
        return [{"role": "user", "content": f"{system_prompt}\n\n{user_message}"}]
    
    elif model_name in ["llama-3.2-1b", "llama-3.2-3b"]:
        return f"""<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    elif model_name == "llama-3-nanda":
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    elif model_name == "llama-3":
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    
    elif model_name == "llama-2":
        return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST]"""
    
    elif model_name == "mistral":
        return f"""<s>[INST] {system_prompt}

{user_message} [/INST]"""
    
    # New branch for DeepSeek-R1-Distill models
    elif model_name.startswith("deepseek"):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_model_terminators(model_name: str, tokenizer) -> List[int]:
    """Gets model-specific termination tokens."""
    common_terminators = [tokenizer.eos_token_id]
    
    terminator_mapping = {
        "llama-3.2-1b": "<|eot_id|>",
        "llama-3.2-3b": "<|eot_id|>",
        "llama-3-nanda": "<|eot_id|>",
        "llama-3": "<|eot_id|>",
        "llama-2": "</s>",
        "mistral": "</s>",
        "gemma-2-9b": "<end_of_turn>",
        "gemma-2-2b": "<end_of_turn>",
    }
    
    if model_name in terminator_mapping:
        return common_terminators + [tokenizer.convert_tokens_to_ids(terminator_mapping[model_name])]
    
    # For DeepSeek models, assume they use a standard <|eot_id|>
    elif model_name.startswith("deepseek"):
        token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if token_id is not None:
            return common_terminators + [token_id]
        else:
            return common_terminators
    return common_terminators

def extract_json_output(generated_text: str) -> str:
    """
    Extracts and returns only the JSON output between <output> and </output> tags.
    If the tags are not found, returns the original text.
    """
    start_tag = "<output>"
    end_tag = "</output>"
    start_idx = generated_text.find(start_tag)
    end_idx = generated_text.find(end_tag)
    if start_idx != -1 and end_idx != -1:
        return generated_text[start_idx + len(start_tag):end_idx].strip()
    return generated_text.strip()

def call_llm(
    model_name: str,
    prompt: Union[str, List[str]],
    temperature: float = None,
    max_new_tokens: int = 32000,
    num_return_sequences: int = 1
) -> List[str]:
    """
    Calls the specified language model to perform cultural adaptation.
    
    Args:
        model_name: Name of the model to use.
        prompt: Input prompt or list of prompts.
        temperature: Sampling temperature (0.0 to 1.0).
        max_new_tokens: Maximum number of tokens to generate.
        num_return_sequences: Number of alternative generations to return.
    
    Returns:
        List of generated responses (final JSON outputs).
    """
    if model_name not in MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    model_id = MODELS[model_name]
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto"
    }
    
    if model_name == "llama-3-nanda":
        model_kwargs["trust_remote_code"] = True

    # Initialize the pipeline
    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs=model_kwargs,
        token=os.environ["HUGGING_FACE_HUB_TOKEN"],
    )

    # Ensure prompt is a list
    if isinstance(prompt, str):
        prompt = [prompt]

    generations = []
    start_time = time.time()

    for i, single_prompt in enumerate(prompt):
        messages = create_prompt(model_name, single_prompt)
        # Optionally, convert messages using the chat template if needed:
        if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            # This assumes the tokenizer has an apply_chat_template method
            formatted_prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            formatted_prompt = messages

        # Get terminators and select the first token as the EOS token (an integer)
        terminators = get_model_terminators(model_name, pipe.tokenizer)
        eos_token_id = terminators[0] if isinstance(terminators, list) and len(terminators) > 0 else None

        # Set sampling parameters for DeepSeek models
        if model_name.startswith("deepseek"):
            if temperature is None:
                temperature = 0.0
            do_sample = False
            top_p = None
        else:
            do_sample = False
            temperature = None
            top_p = None

        try:
            outputs = pipe(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                pad_token_id=pipe.tokenizer.eos_token_id,
                eos_token_id=eos_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                return_full_text=False,
            )

            for output in outputs:
                raw_text = output['generated_text'].strip()
                # Post-process to extract only the JSON output
                json_output = extract_json_output(raw_text)
                generations.append(json_output)

        except Exception as e:
            print(f"Error processing prompt {i}: {str(e)}")
            generations.append(None)

    elapsed_time = time.time() - start_time
    print(f"Generation completed in {elapsed_time:.2f} seconds")
    
    return generations
