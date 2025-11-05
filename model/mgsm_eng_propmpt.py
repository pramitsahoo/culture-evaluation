import os
import csv
import json
import time
import torch
import transformers
import multiprocessing

# Set your Hugging Face Hub token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HUGGING_FACE_HUB_TOKEN"

# Define models (excluding "llama-3-nanda")
MODELS = {
    "llama-3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma-2-2b": "google/gemma-2-2b-it",
}

# Assign each model to a GPU (adjust as needed; here GPUs 0 through 6 are used)
MODEL_GPU_MAPPING = {
    "llama-3": 0,
    "llama-2": 1,
    "mistral": 2,
    "llama-3.2-1b": 3,
    "llama-3.2-3b": 4,
    "gemma-2-9b": 5,
    "gemma-2-2b": 6,
}

# Define the system prompt for cultural adaptation (inline)
system_prompt = (
    """You are an AI assistant tasked with adapting a text to suit a specific cultural context in a particular language while maintaining the original meaning and intent. Your goal is to ensure the text feels natural and appropriate for the target audience, considering cultural nuances, values, and sensitivities. When making these adaptations, follow these key steps:

1. Cultural Relevance: Adjust any idioms, metaphors, or cultural references that may not resonate with the target audience. Replace them with culturally appropriate alternatives.
2. Tone and Intent: Preserve the original emotional tone and message, even when making cultural adjustments.
3. Cultural Sensitivity: Be mindful of topics, words, or phrases that might be sensitive or inappropriate in the target culture.

Specific guidelines:
- Replace foreign names with common Indian names (female for female, male for male).Use a diverse set of names.
- Use Indian locations in place of foreign locations.
- Convert all foreign currencies to Indian Rupees (INR) using a clear symbolic or approximate rate (for example, $1 = ₹83). Remove any remaining references to foreign currency (USD, etc.).
- Incorporate Indian traditions, festivals, and cultural practices only if it is contextually appropriate and does not distort the original meaning.
- Use regional-specific terminology and expressions without changing the logical sense of the text.
- Replace foreign food items with Indian equivalents only if it makes sense. For example, “muffins” can become “parathas,” but do not replace food items that are already commonly used in India or are essential to the text’s logic (e.g., do not replace “eggs” if it’s about chickens laying eggs).
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

Example of correct replaced_concepts:
Good: {"John":"Ramesh", "muffins":"parathas", "$10":"₹830"}
Bad: {"eggs":"eggs", "John":"Ramesh", "muffins":"parathas", "$10":"\\u20b9830"}

Provide only the adapted text and replaced words of the problem statement in a valid JSON format, like this example:
[{"cultural_adapted_text":"Ramesh bought 5 parathas for ₹830 and gave 2 to his friend. How much money did he spend per paratha?","replaced_concepts":{"John":"Ramesh","muffins":"parathas","$10":"₹830"}}]"""
)

# Read input prompts from "mgsm_bn.tsv" (first column only)
input_prompts = []
with open("/u/student/2023/ai23mtech14004/culture-evaluation/MGSM/mgsm_bn.tsv", "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        if row and row[0].strip():
            input_prompts.append(row[0].strip())

if not input_prompts:
    print("No input prompts found in the TSV file.")
    exit()

def process_model(model_name, model_id, gpu_id, input_prompts, system_prompt):
    print(f"\nProcessing model: {model_name} on GPU: {gpu_id}")
    # Set model parameters (load on the specified GPU if given)
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if gpu_id is not None:
        model_kwargs["device_map"] = {"": gpu_id}
    else:
        model_kwargs["device_map"] = "auto"

    # Initialize the text-generation pipeline
    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs=model_kwargs,
        token=os.environ["HUGGING_FACE_HUB_TOKEN"],
    )

    # List to store outputs for the current model
    model_outputs = []

    # Process each input prompt
    for prompt in input_prompts:
        # Build the user message
        user_message = f"Adapt the following text to Indian culture and provide the output in the specified single-line format:\n{prompt}"
        
        # Create the final prompt based on model type
        if model_name in ["gemma-2-9b", "gemma-2-2b"]:
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_message}"}]
        elif model_name in ["llama-3.2-1b", "llama-3.2-3b"]:
            messages = (
                f"<|start_header_id|>system<|end_header_id|>\n"
                f"{system_prompt}\n"
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
        elif model_name == "llama-3":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        elif model_name == "llama-2":
            messages = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"
        elif model_name == "mistral":
            messages = f"<s>[INST] {system_prompt}\n\n{user_message} [/INST]"
        else:
            messages = f"{system_prompt}\n\n{user_message}"
        
        # Determine termination tokens based on model type
        common_terminators = [pipe.tokenizer.eos_token_id]
        terminator = None
        if model_name in ["llama-3.2-1b", "llama-3.2-3b", "llama-3"]:
            terminator = "<|eot_id|>"
        elif model_name in ["llama-2", "mistral"]:
            terminator = "</s>"
        elif model_name in ["gemma-2-9b", "gemma-2-2b"]:
            terminator = "<end_of_turn>"
        if terminator:
            term_token_id = pipe.tokenizer.convert_tokens_to_ids(terminator)
            eos_token_ids = common_terminators + [term_token_id]
        else:
            eos_token_ids = common_terminators

        # Generate the output
        try:
            outputs = pipe(
                messages,
                max_new_tokens=2048,
                pad_token_id=pipe.tokenizer.eos_token_id,
                eos_token_id=eos_token_ids,
                do_sample=False,
                temperature=0.0,
                top_p=None,
                num_return_sequences=1,
                return_full_text=False,
            )
            if outputs and "generated_text" in outputs[0]:
                output_text = outputs[0]["generated_text"].strip()
            else:
                output_text = None
        except Exception as e:
            print(f"Error processing prompt:\n{prompt}\nError: {e}")
            output_text = None

        model_outputs.append(output_text)
    
    # Save outputs for the current model into a JSON file
    output_filename = f"output_{model_name}.json"
    with open(output_filename, "w", encoding="utf-8") as out_f:
        json.dump(model_outputs, out_f, ensure_ascii=False, indent=2)
    print(f"Results for model '{model_name}' saved to {output_filename}")

if __name__ == "__main__":
    processes = []
    # Spawn a separate process for each model so that they run concurrently on different GPUs.
    for model_name, model_id in MODELS.items():
        gpu_id = MODEL_GPU_MAPPING.get(model_name, None)
        p = multiprocessing.Process(target=process_model, args=(model_name, model_id, gpu_id, input_prompts, system_prompt))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete.
    for p in processes:
        p.join()
    
    print("All models have been processed in parallel.")
