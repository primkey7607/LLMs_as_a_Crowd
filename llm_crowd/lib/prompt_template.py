class PromptTemplate:
    def __init__(
            self,
            name: str,
            system: str,
            preamble: str,
            sentence1: str,
            sentence2: str,
            question: str,
    ):
        self.name = template_name
        self.system = system
        self.preamble = preamble
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.question = question

    def get_response(
            self,
            left: str,
            right: str,
            examples: List[Tuple[str, str, Any]] = []) -> Tuple[str, Any]:
        '''
        Return the raw response and the parsed answer
        '''
        pass

    def get_example_answer(label: Any)
        '''
        Return the desired string answer for a few-shot example with the given label
        '''
        pass

    def get_prompt(
            self,
            left: Any,
            right: Any,
            examples: List[Tuple[str, str, Any]] = []) -> List[Dict[str, str]]:
        '''
        Return a full prompt, possibly with few-shot examples, in OpenAI chat format
        '''
        chat = [{'role': 'system', 'content': system}]

        for ex_left, ex_right, label in examples:
            chat.append({'role': 'user', 'content': self.get_base_prompt(ex_left, ex_right)})
            chat.append({'role': 'assistant', 'content': self.get_example_answer(label)})

        chat.append({'role': 'user', 'content': self.get_base_prompt(left, right)})
        # Add preamble to first base prompt, whether an example or the actual prompt
        chat[1]['content'] = self.preamble + '\n\n' + chat[1]['content']
        return chat

    def get_base_prompt(self, left, right):
        '''
        Return basic string prompt without preamble or few-shot examples
        '''
        out = [
            self.sentence1, left, '',
            self.sentence2, right, '',
            self.question,
        ]
        return '\n'.join(out)
