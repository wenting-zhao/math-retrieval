import os
import time
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def chat_compeletion_openai(model, messages, temperature=1, max_tokens=512):
    error = True
    while error:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content.strip()
            error = False
        except openai._exceptions.OpenAIError as e:
            time.sleep(5)
            print(type(e), e)
    return output

