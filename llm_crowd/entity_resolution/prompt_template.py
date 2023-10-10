from typing import Any, List, Optional, Tuple

from llm_crowd.lib.llm import llm
from llm_crowd.lib.prompt_template import PromptTemplate

TEMPLATES = {
    'baseline': {
        'preamble': "We are trying to integrate product data from two different databases. The goal is to look at two product entries, one from each database, and determine whether the two entries refer to the same product or not. Since the databases are different, there will still be some differences between entries that refer to the same product.",
        'sentence1': "Here is an entry from the first database:",
        'sentence2': "Here is an entry from the second database:",
        'question': "As best as you can tell, do these entries refer to the same product?",
    },
    # Note: this prompts the opposite answer - "YES" means no match.
    'customer': {
        'preamble': "Suppose you are an employee who works at the customer support division of a large company. You receive a complaint from a customer demanding a refund because the product label information did not match the actual product they received.",
        'sentence1': "This is the product label information in the customer's claim:",
        'sentence2': "Upon inspection of the product the customer purchased, you find the following information:",
        'question': "Do you owe the customer a refund?",
    },
    'detective': {
        'preamble': "Suppose you're a detective who has been hired to detect illegal copies of products so that law enforcement can fine those selling these copies.",
        'sentence1': "Suppose you are given the following information about a product that a business may be trying to sell illegally:",
        'sentence2': "You confront the storefront owner potentially selling the illegal copy. You find the following information:",
        'question': "Based on the available information, is it likely that the storefront owner is selling an illegal copy of the product?",
    },
    'journalist': {
        'preamble': "I'm a journalist doing a piece on the dwindling importance of product descriptions in distinguishing two products without seeing them in-person.",
        'sentence1': "As an example, I wanted to have the following product descriptions that might be different, but represent the same product:",
        'sentence2': "and",
        'question': "Is this an effective example of product descriptions that might look different, but describe the same product?",
    },
    'layperson': {
        'preamble': "I'm a father with a kid for whom I'm trying to find a gift.",
        'sentence1': "I've been wandering around the store all day looking for the following product for my kid:",
        'sentence2': "I found a product that seems like what I want:",
        'question': "But I can't tell. Is this the product I'm looking for?",
    },
    'plain': {
        'preamble': "Consider the following pair of csv file rows:",
        'sentence1': "",
        'sentence2': "and",
        'question': "Can you confirm if this pair of rows is a match after performing entity resolution on two tables?",
    },
    'security': {
        'preamble': "I am a computer security expert for a large e-commerce company. It's very common for our users to spoof pages of existing products to scam unsuspecting customers. It's killing our business!",
        'sentence1': "I found two product pages with the following product information:",
        'sentence2': "and",
        'question': "Are these pages talking about the same product? If so, one of them is spoofed...",
    },
    'veryplain': {
        'preamble': "",
        'sentence1': "",
        'sentence2': "###",
        'question': "Same product?",
    },
}

class ERPromptTemplate(PromptTemplate):
    def __init__(self, template_name: str, cot: bool, temperature: float):
        if template_name not in TEMPLATES:
            raise ValueError(f"Unrecognized template name {template_name}")
    
        followup = "Please answer in a single word (YES or NO). If you are uncertain, make your best guess."
        if cot:
            system = "You are a helpful assistant who thinks step-by-step and then gives final yes or no answers."
            question_ending = "\n\nAnswer: Let's think step-by-step."
            followup = "What is your final answer? " + followup
        else:
            system = "You are a helpful assistant who can only answer YES or NO and then explain your reasoning."
            question_ending = " Begin your answer with YES or NO."

        template = TEMPLATES[template_name]
        super().__init__(
            template_name,
            system,
            template['preamble'],
            template['sentence1'],
            template['sentence2'],
            template['question'] + question_ending,
        )
        
        self.cot = cot
        self.temperature = temperature
        self.flipped = (template_name == 'customer')

    def get_response(
            self,
            model: str,
            left: str,
            right: str,
            examples: List[Tuple[str, str, Any]] = []) -> Tuple[str, Any]:
        prompt = self.get_prompt(left[:1000], right[:1000], examples)
        max_tokens = 300 if self.cot else 30
        response = llm(model, prompt, self.temperature, max_tokens)
        answer = self.parse_response(response)
        if answer is None:
            prompt.append({'role': 'assistant', 'content': response})
            followup = "Please answer in a single word (YES or NO). If you are uncertain, make your best guess."
            if self.cot:
                followup = f"What is your final answer? {followup}"
            prompt.append({'role': 'user', 'content': followup})
            response2 = llm(model, prompt, self.temperature, 5)
            answer = self.parse_response(response2)
        return (response, answer)

    def get_example_answer(label: Any):
        if label ^ self.flipped:
            return "YES."
        return "NO."

    def parse_response(self, response: str) -> Optional[int]:
        answer = None
        if self.cot:
            if 'yes' in response.lower()[-6:]:
                answer = 1
            elif 'no' in response.lower()[-6:]:
                answer = 0
        else:
            if response.lower().startswith('yes'):
                answer = 1
            elif response.lower().startswith('no'):
                answer = 0
        if self.flipped and answer is not None:
            answer = 1 - answer
        return answer
