

######################################################################################
import torch
import transformers
import time
from typing import Union, List
from huggingface_hub import login
import os

# Set your Hugging Face Hub token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HUGGING_FACE_HUB_TOKEN"

# Define the models with their corresponding Hugging Face identifiers
MODELS = {
    "llama-3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3-nanda": "MBZUAI/Llama-3-Nanda-10B-Chat",
    "gemma-2": "google/gemma-2-9b-it",
    "gemma-2-2b": "google/gemma-2-2b-it",

}



def get_system_prompt():
    """সাংস্কৃতিক অভিযোজন জন্য সিস্টেম প্রম্পট"""
    return """আপনি একজন এআই সহকারী যার কাজ মূল অর্থ ও উদ্দেশ্য বজায় রেখে টেক্সটকে বাংলা ভাষা ও সংস্কৃতির সাথে খাপ খাইয়ে দেওয়া। আপনার লক্ষ্য টেক্সটটিকে বাংলাভাষী শ্রোতাদের জন্য স্বাভাবিক ও প্রাসঙ্গিক করে তোলা, সাংস্কৃতিক সূক্ষ্মতা, মূল্যবোধ ও সংবেদনশীলতা বিবেচনা করে। নিচের ধাপগুলি অনুসরণ করুন:

১. সাংস্কৃতিক প্রাসঙ্গিকতা: বিদেশি প্রবাদ, রূপক বা সাংস্কৃতিক উল্লেখগুলিকে বাংলার সমতুল্য দিয়ে প্রতিস্থাপন করুন
২. ভাষারীতি: সহজ, প্রমিত বাংলা ব্যবহার করুন (সাধু ভাষা) আঞ্চলিক উপভাষা ছাড়া
৩. সাংস্কৃতিক সংবেদনশীলতা: বাংলার সাংস্কৃতিক রীতিনীতি মেনে চলুন

নির্দেশাবলী:
- বিদেশি নামগুলিকে সাধারণ বাংলা নামে পরিবর্তন করুন (মহিলা/পুরুষের জন্য আলাদা)
- বিদেশি স্থানগুলিকে বাংলার স্থাননাম ব্যবহার করুন
- মুদ্রাকে ভারতীয় রুপিতে (₹) রূপান্তর করুন (যেমন: $১ = ₹৮২)
- বাংলার উৎসব/প্রথা যোগ করুন যেখানে প্রাসঙ্গিক
- পশ্চিমা সংখ্যার (123) বদলে বাংলা সংখ্যা ব্যবহার করুন (১২৩)
- গাণিতিক অপারেশন ও সংখ্যাসমূহ অপরিবর্তিত রাখুন
- উত্তর অবশ্যই বাংলা স্ক্রিপ্টে দিতে হবে
- সিঙ্গেল-লাইন JSON অ্যারে ফরম্যাটে উত্তর দিন

উদাহরণ:
[{"cultural_adapted_text": "রমেশ ৫টি পরোটা কিনলেন ৮৩০ টাকায় এবং ২টি বন্ধুকে দিলেন। প্রতি পরোটার দাম কত?","replaced_concepts":{"John":"রমেশ","muffins":"পরোটা","$10":"৮৩০ টাকা"}}]"""

def create_prompt(model_name: str, user_prompt: str) -> Union[str, List[dict]]:
    """মডেল-স্পেসিফিক বাংলা প্রম্পট তৈরি"""
    system_prompt = get_system_prompt()
    user_message = f"নিচের টেক্সটটি বাংলা ভাষা ও সংস্কৃতিতে রূপান্তর করুন এবং JSON ফরম্যাটে উত্তর দিন:\n{user_prompt}"

    if model_name == "gemma-2":
        return [{"role": "user", "content": f"{system_prompt}\n\n{user_message}"}]
    elif model_name == "gemma-2-2b":
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
    
    else:
        raise ValueError(f"অসমর্থিত মডেল: {model_name}")

# টার্মিনেশন টোকেন ও অন্যান্য ফাংশন পূর্বের মতোই থাকবে

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
        "gemma-2": "<end_of_turn>" ,
        "gemma-2-2b": "<end_of_turn>",
    }
    
    if model_name in terminator_mapping:
        return common_terminators + [tokenizer.convert_tokens_to_ids(terminator_mapping[model_name])]
    return common_terminators

def call_llm(
    model_name: str,
    prompt: Union[str, List[str]],
    temperature: float = None,
    top_p:float = None,
    max_new_tokens: int = 4096,
    num_return_sequences: int = 1
) -> List[str]:
    """
    Calls the specified language model to perform cultural adaptation.
    
    Args:
        model_name: Name of the model to use
        prompt: Input prompt or list of prompts
        temperature: Sampling temperature (0.0 to 1.0)
        max_new_tokens: Maximum number of tokens to generate
        num_return_sequences: Number of alternative generations to return
    
    Returns:
        List of generated responses
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

    # Initialize the pipeline without device_map in pipeline args
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
        terminators = get_model_terminators(model_name, pipe.tokenizer)

        try:
            outputs = pipe(
                messages,
                max_new_tokens=max_new_tokens,
                pad_token_id=pipe.tokenizer.eos_token_id,
                eos_token_id=terminators,
                do_sample=False,
                temperature=None,
                top_p=None,
                num_return_sequences=num_return_sequences,
                return_full_text=False,
            )

            for output in outputs:
                generated_text = output['generated_text'].strip()
                generations.append(generated_text)

        except Exception as e:
            print(f"Error processing prompt {i}: {str(e)}")
            generations.append(None)

    elapsed_time = time.time() - start_time
    print(f"Generation completed in {elapsed_time:.2f} seconds")
    
    return generations