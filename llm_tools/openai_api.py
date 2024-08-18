import sys
from openai import OpenAI


class ChatGPTTool(object):
    def __init__(self):
        self.API_SECRET_KEY = "your_api_secret_key"
        self.BASE_URL = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=self.API_SECRET_KEY, base_url=self.BASE_URL)

    def chat(self, input_text):
        resp = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": input_text}
            ],
            temperature=0.0,
            seed=42,
        )

        resp_msg = resp.msg
        if resp_msg == 'error':
            print('---------------------- error info ----------------------')
            print('error: ', resp.error['message'])
            print('code: ', resp.error['code'])
            print('---------------------- error info ----------------------\n\n')
            sys.exit(1)

        output = resp.choices[0].message.content
        return output

    def chat_with_context(self, messages):
        resp = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0,
            seed=42,
        )

        resp_msg = resp.msg
        if resp_msg == 'error':
            print('---------------------- error info ----------------------')
            print('error: ', resp.error['message'])
            print('code: ', resp.error['code'])
            print('---------------------- error info ----------------------\n\n')
            sys.exit(1)

        output = resp.choices[0].message.content
        return output
