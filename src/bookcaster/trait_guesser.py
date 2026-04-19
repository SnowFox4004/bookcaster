from openai import OpenAI
import json
from .llms import LLMProvider
from .prompts import AGE_GENDER_TRAIT_PROMPT
from .utils import Chapter
from loguru import logger


class TraitGuesser:

    async def guess(self, text: str, characters: list, *args, **kwargs) -> list:
        pass


class AgeGenderGuesser(TraitGuesser):
    """
    use llm to guess the age and gender of a character, then generate voice prompts
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def guess(self, text: str, characters: list) -> list:
        """
        guess the age and gender of a character
        """
        # characters = set(speech.get("speaker") for speech in chapter.script)
        logger.info(f'Guessing age and gender for {", ".join(characters)}')
        response = await self.llm.generate(
            AGE_GENDER_TRAIT_PROMPT.format(text=text, characters=", ".join(characters)),
            schema={
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "array",
                        "properties": {
                            "character": {"type": "string"},
                            "age": {"type": "string"},
                            "gender": {"type": "string"},
                        },
                    }
                },
            },
        )
        response = json.loads(response)

        for trait in response:
            trait["voice_prompt"] = trait["age"] + trait["gender"] + "的声音"

        return response
