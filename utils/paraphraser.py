import os
import time
import google.generativeai as genai
import google.api_core.exceptions
from dotenv import load_dotenv

# Load keys from .env
load_dotenv()
key_env = ['GEMINI', 'GEMINI2', 'GEMINI3', 'GEMINI4']
api_keys = [os.getenv(k) for k in key_env if os.getenv(k)]

# System prompt
sys_prompt = """
You are MedPara-Bot, an expert-level medical paraphrasing assistant.
When given a sentence containing medical terminology:
  • Preserve all clinical meaning and technical accuracy.
  • Rewrite the sentence in clear, fluent English.
  • Maintain any critical lab values, units, drug names, or anatomy terms exactly.
  • Do not add or remove medical facts or change the diagnostic intent.
  • Avoid additional commentary or explanations, anything outside the sentence.
  • Do not use the phrase "paraphrase" or "paraphrased" in your response.
Return only the paraphrased sentence—no explanations or extra text.
"""

# Rotate through API keys until one works
def rotate_api_key(keys, start_index=0):
    num_keys = len(keys)
    for i in range(num_keys):
        idx = (start_index + i) % num_keys
        try:
            genai.configure(api_key=keys[idx])
            model = genai.GenerativeModel('gemini-1.5-pro')
            return model, idx
        except Exception as e:
            print(f"Failed to configure key {keys[idx][:6]}...: {e}")
            continue
    raise Exception("All API keys failed to configure.")

# Call the model with retries and key rotation
def paraphrase_model(model, input_text, keys, current_key_index):
    retry_delay = 1
    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                contents=sys_prompt + "\n\nPatient sentence: " + input_text + "\nParaphrase:",
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=256
                )
            )
            return response.text, model, current_key_index

        except google.api_core.exceptions.ResourceExhausted as e:
            if e.code == 429:
                print(f"Rate limit on key {[current_key_index]}..., rotating key.")
                model, current_key_index = rotate_api_key(keys, current_key_index + 1)
                continue
            else:
                raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception("Failed to paraphrase after multiple retries.")
    
    raise Exception("Paraphrasing failed completely.")

def paraphrase(input_text: str) -> str:
    model, key_index = rotate_api_key(api_keys)
    result, _, _ = paraphrase_model(model, input_text, api_keys, key_index)
    return result
