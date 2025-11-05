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
    "llama-3": 1,
    "llama-2": 2,
    "mistral": 3,
    "llama-3.2-1b": 4,
    "llama-3.2-3b": 5,
    "gemma-2-9b": 6,
    "gemma-2-2b": 15,
}

# Define the system prompt for cultural adaptation (inline)
system_prompt = (
"""আপনি একজন এআই সহকারী যার কাজ মূল অর্থ ও উদ্দেশ্য বজায় রেখে টেক্সটকে বাংলা ভাষা ও সংস্কৃতির সাথে খাপ খাইয়ে দেওয়া। আপনার লক্ষ্য টেক্সটটিকে বাংলাভাষী শ্রোতাদের জন্য স্বাভাবিক ও প্রাসঙ্গিক করে তোলা, সাংস্কৃতিক সূক্ষ্মতা, মূল্যবোধ ও সংবেদনশীলতা বিবেচনা করে। নিচের ধাপগুলি অনুসরণ করুন:

১. সাংস্কৃতিক প্রাসঙ্গিকতা: বিদেশি প্রবাদ, রূপক বা সাংস্কৃতিক উল্লেখগুলিকে বাংলার সমতুল্য দিয়ে প্রতিস্থাপন করুন
২. ভাষারীতি: সহজ, প্রমিত বাংলা ব্যবহার করুন (সাধু ভাষা) আঞ্চলিক উপভাষা ছাড়া
৩. সাংস্কৃতিক সংবেদনশীলতা: বাংলার সাংস্কৃতিক রীতিনীতি মেনে চলুন

নির্দেশাবলী:
- বিদেশি নামগুলিকে সাধারণ বাংলা নামে পরিবর্তন করুন (মহিলা/পুরুষের জন্য আলাদা)
- বিদেশি স্থানগুলিকে বাংলার স্থাননাম ব্যবহার করুন
- মুদ্রাকে বাংলাদেশী টাকায় (৳) রূপান্তর করুন (যেমন: $১ = ৳১১৮)
- বাংলার উৎসব/প্রথা যোগ করুন যেখানে প্রাসঙ্গিক
- পশ্চিমা সংখ্যার (123) বদলে বাংলা সংখ্যা ব্যবহার করুন (১২৩)
- গাণিতিক অপারেশন ও সংখ্যাসমূহ অপরিবর্তিত রাখুন
- উত্তর অবশ্যই বাংলা স্ক্রিপ্টে দিতে হবে
- সিঙ্গেল-লাইন JSON অ্যারে ফরম্যাটে উত্তর দিন

উদাহরণ:
[{"cultural_adapted_text": "রমেশ ৫টি পরোটা কিনলেন ৮৩০ টাকায় এবং ২টি বন্ধুকে দিলেন। প্রতি পরোটার দাম কত?","replaced_concepts":{"John":"রমেশ","muffins":"পরোটা","$10":"৮৩০ টাকা"}}]"""


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
    output_filename = f"bn_output_{model_name}.json"
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
