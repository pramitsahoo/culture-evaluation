import torch
import transformers
import os

# Set your Hugging Face Hub token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HUGGING_FACE_HUB_TOKEN"

# Define the models with their corresponding Hugging Face identifiers
models = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3-nanda": "MBZUAI/Llama-3-Nanda-10B-Chat"  
}

def create_prompt(model_name: str, user_prompt: str):
#     base_instruction = """
# Adapt the given text to reflect contemporary Indian culture in, settings, and context, while preserving the original mathematical problem and numerical values.
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
################################################################################################
#                   For Indian English                                                         #
################################################################################################
#     base_instruction = """
# Adapt the given text to reflect contemporary Indian culture in settings, and context, while preserving the original mathematical problem and numerical values.
# Rewrite the text in Indian English, incorporating these linguistic characteristics:
# - Use typical Indian English expressions and colloquialisms
# - Incorporate Indian English sentence structures
# - Add typical Indian English speech patterns
# - Use regional-specific vocabulary and idioms
# - Employ characteristic Indian English intonations and phrasings

# Specific cultural adaptation guidelines:
# - Replace names with common Indian names (female for female, male for male)
# - Use Indian locations
# - Convert currencies to Indian Rupees (INR)
# - Remove foreign currency references (USD, etc.)
# - Incorporate Indian traditions, festivals, and cultural practices
# - Use regional-specific terminology and expressions
# - Replace foreign food items with Indian equivalents (e.g., muffins to parathas)
# - Maintain original mathematical operations and numerical values

# Indian English Language Characteristics to Include:
# - Use of "no" for emphasis (e.g., "It's good, no?")
# - Redundant pronouns (e.g., "I myself will do it")
# - Tag questions (e.g., "You are coming, na?")
# - Specific sentence constructions like "What to do?" or "Doing timepass"
# - Direct translations of local language idioms

# Provide only the adapted text and replaced words of the problem statement in a valid json format just like this example - 
# [
#     {
#         "adapted_text": "Ramesh only bought 5 parathas for ₹830, no? He gave 2 to his friend. Tell me, how much money did he spend per paratha, yaar?",
#         "replaced_words": {"John": "Ramesh", "muffins": "parathas", "$10": "₹830"}
#     }
# ]
# Do not include any calculations, step-by-step solutions, or explanations.
# Do not solve the problem or provide the answer.
# Do not hallucinate.
# """


################################################################################################
#                   For Bengali                                                         #
################################################################################################

    base_instruction = """
Adapt the given text to reflect Bengali culture, language, and context, while preserving the original mathematical problem and numerical values.
Specific guidelines for Bengali language and cultural adaptation:

Language Characteristics:
- Use authentic Bengali grammatical structures
- Incorporate typical Bengali linguistic patterns
- Preserve Bengali sentence construction (Subject-Object-Verb order)
- Use appropriate Bengali honorifics and formal/informal language
- Include characteristic Bengali expressions and colloquialisms

Cultural Adaptation Requirements:
- Replace names with common Bengali names (e.g., Arnab, Suchitra, Riya)
- Use Bengali locations (cities like Kolkata, Howrah, or specific districts)
- Convert currencies to Indian Rupees (₹)
- Incorporate Bengali cultural references
- Use local Bengali terminology and expressions
- Reference Bengali traditions, festivals (like Durga Puja, Pohela Boishakh)
- Replace foreign food items with Bengali equivalents (e.g., muffins to rosogulla)
- Maintain original mathematical operations and numerical values

Bengali Language Nuances to Include:
- Use of respectful and contextual verb forms
- Appropriate use of postpositions
- Incorporation of Bengali-specific metaphors and idioms
- Reflect the musical and poetic quality of Bengali language
- Use of contextual honorifics (e.g., "da" for elder brother, "di" for elder sister)

Output Format:
Provide the adapted text and replaced words in a valid JSON format:
[
    {
        "adapted_text": "অর্ণব ৫টি রসগোল্লা কিনেছিল ৮৩০ টাকায়। সে তার বন্ধুকে ২টি দিয়েছিল। প্রতি রসগোল্লায় কত টাকা খরচ হয়েছে?",
        "replaced_words": {
            "John": "অর্ণব",
            "muffins": "রসগোল্লা", 
            "$10": "৮৩০ টাকা"
        },
        "transliteration": "Arnab 5ti rosogolla kinechilo 830 takay. Se tar bondhuke 2ti diyechilo. Prati rosogollay koto taka khoroch hoyechilo?"
    }
]

Important Notes:
- Preserve the original mathematical problem's numerical values
- Maintain the core problem's structure
- Do not solve the problem
- Provide authentic Bengali language representation
- Do not hallucinate additional information
"""

    user_example = "John bought 5 muffins for $10 and gave 2 to his friend. How much money did he spend per muffin?"

    # Print the prompt for each model type
    print(f"\n--- Prompt for {model_name} ---")
    
    if model_name in ["llama-3.2-1b", "llama-3.2-3b"]:
        prompt = f"""<|start_header_id|>system<|end_header_id|>
{base_instruction}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_example}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        print(prompt)
        return prompt
    
    elif model_name == "llama-3-nanda":
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>{user_example}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        print(prompt)
        return prompt
    
    elif model_name == "llama-3":
        prompt = [
            {"role": "system", "content": base_instruction},
            {"role": "user", "content": user_example},
        ]
        print("System Prompt:", prompt[0]['content'])
        print("User Prompt:", prompt[1]['content'])
        return prompt
    
    elif model_name == "llama-2":
        prompt = f"""<s>[INST] <<SYS>>
{base_instruction}
<</SYS>>

{user_example} [/INST]
"""
        print(prompt)
        return prompt
    
    elif model_name == "mistral":
        prompt = f"""<s>[INST] {base_instruction}

{user_example} [/INST]
"""
        print(prompt)
        return prompt
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# Iterate through and print prompts for all models
for model_name in models.keys():
    create_prompt(model_name, "")