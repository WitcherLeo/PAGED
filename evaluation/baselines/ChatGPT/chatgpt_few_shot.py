from evaluation.baselines.prompt import get_prompt
from llm_tools.openai_api import ChatGPTTool


class PEChatGPTFew(object):
    def __init__(self):
        self.tool = ChatGPTTool()

    def batch_predict(self, paragraphs: list):
        prompt = get_prompt()
        input_texts = [prompt.format(paragraph) for paragraph in paragraphs]

        outputs = []
        for input_text in input_texts:
            output = self.tool.chat(input_text)
            outputs.append(output)
        return outputs
