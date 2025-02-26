import os
import platform
from typing import Optional
import torch
from huggingface_hub import login
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()


def prompt(prompt: str, type="API", model="gpt-4o-mini") -> str:
    """Prompt the AI with a given text and return the response.

    Args:
        prompt (str): The text to prompt the AI with.
        type (str, optional): API or local execution. Defaults to "API".
        model (str, optional): The model to use for the AI response, either an openAI or huggingface identifier. Defaults to "gpt-4o-mini".

    Returns:
        str: The response from the AI.
    """

    match type:
        case "API":
            client = OpenAI(
                api_key=os.getenv("OPENAI_KEY")
            )

            completion = client.chat.completions.create(
                model=model,
                store=True,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return str(completion.choices[0].message.content)
    
        case "LOCAL":
            device = (
                "mps" if platform.system() == "Darwin" and torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else
                "cpu"
            )

            tokenizer = AutoTokenizer.from_pretrained(model)
            model = AutoModelForCausalLM.from_pretrained(
                model, device_map="auto" if device != "cpu" else None
            ).to(device)

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            output_ids = model.generate(input_ids, max_length=50)

            return tokenizer.decode(output_ids[0], skip_special_tokens=True)
        case _:
            raise ValueError("Invalid type argument. Use either 'API' or 'LOCAL'.")