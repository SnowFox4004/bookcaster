from openai import OpenAI, AsyncOpenAI, omit
import os


class LLMProvider:
    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def generate(self, prompt: str, schema=omit) -> str:

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.Follow the user's instructions carefully especially the formats.dont generate anything that is fake.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            response_format=schema,
        )

        return response.choices[0].message.content
