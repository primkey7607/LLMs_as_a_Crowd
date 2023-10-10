from typing import Dict, List

import openai
import signal
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

class MyTimeoutException(Exception):
    pass

#register a handler for the timeout
def handler(signum, frame):
    print("Waited long enough!")
    raise MyTimeoutException("STOP")

def llm(model, chat, temperature, max_tokens=30):
    return LLMS[model](chat, temperature, max_tokens=max_tokens)
    

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatgpt(chat: List[Dict[str, str]], temperature: float, max_tokens: int = 30) -> str:
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(30)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=chat,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"]

_llama_model = None
_llama_tokenizer = None

def openllama(chat: List[Dict[str, str]], temperature: float, max_tokens: int = 30) -> str:
    load_llama()
    global _llama_model
    global _llama_tokenizer

    prompt = convert_chat_to_prompt(chat)
    input_ids = _llama_tokenizer(prompt, return_tensors="pt").input_ids

    generation_output = _llama_model.generate(
        input_ids=input_ids, max_new_tokens=max_tokens, temperature=temperature
    )
    return _llama_tokenizer.decode(generation_output[0])

ANSWER_STRING = "\n\n-----\n\nAnswer:\n\n"

def convert_chat_to_prompt(chat: List[Dict[str, str]]) -> str:
    prompt = ""
    for item in chat:
        if item['role'] == 'system':
            continue
        elif item['role'] == 'user':
            prompt += item['content']
            prompt += ANSWER_STRING
        elif item['role'] == 'assistant':
            prompt += item['content']
    return prompt

def load_llama():
    global _llama_model
    global _llama_tokenizer
    if _llama_model is not None:
        return
    ## v2 models
    model_path = 'openlm-research/open_llama_3b_v2'
    # model_path = 'openlm-research/open_llama_7b_v2'

    ## v1 models
    # model_path = 'openlm-research/open_llama_3b'
    # model_path = 'openlm-research/open_llama_7b'
    # model_path = 'openlm-research/open_llama_13b'

    _llama_tokenizer = LlamaTokenizer.from_pretrained(model_path)
    _llama_model = LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map='auto',
    )

LLMS = {
    'chatgpt': chatgpt,
    'llama': openllama,
}
