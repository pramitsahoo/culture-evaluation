import os
from .offline_copy import call_llm

# standard_prompt = """
#     Imagine you are an Indian person. Convert the text to Indian Culture.
    
#     Text: {input}     
#     """

standard_prompt = """Text: {input}"""

def prompt_model(input_text, model_name, prompt="standard", temperature=0.7, max_new_tokens=256, batch=True):
    # print(input_text)
    if prompt == "standard":
        if batch:
            formatted_prompt = []
            for inp in input_text:
                # print(standard_prompt.format(input=inp))
                formatted_prompt.append(standard_prompt.format(input=inp))
        else:
            formatted_prompt = standard_prompt.format(input=input_text)
    else:
        NotImplementedError
    # print("Formatted prompt: ", formatted_prompt)

    return call_llm(model_name, formatted_prompt, temperature=temperature, max_new_tokens=max_new_tokens)
