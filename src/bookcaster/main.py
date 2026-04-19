import os
import aiofiles
from dataclasses import dataclass
import asyncio
import json
from loguru import logger
from tqdm import tqdm

from bookcaster.utils import Chapter
from bookcaster.llms import LLMProvider
from bookcaster.trait_guesser import AgeGenderGuesser
from bookcaster.prompts import *


class Bookcaster:

    def _init_default_providers(self):
        if self.providers.get("llm", None) is None:
            self.providers["llm"] = LLMProvider(
                model="qwen3.5-4B",
                api_key="lmstudio",
                base_url="http://127.0.0.1:1234/",
            )

        if self.providers.get("trait_guesser", None) is None:
            self.providers["trait_guesser"] = AgeGenderGuesser(
                llm=self.providers["llm"],
            )

    def __init__(
        self,
        path: str,
        providers: dict[str, str] = None,
    ):

        self.path = path
        self.chapters = []
        self.providers = providers
        self.character_traits = {}

        self._init_default_providers()

    async def podcast(self):
        logger.info("Start podcasting")

        # read all chapters
        for idx, chapter in enumerate(os.listdir(self.path)):
            if not chapter.endswith(".txt"):
                continue

            async with aiofiles.open(
                os.path.join(self.path, chapter), "r", encoding="utf-8"
            ) as f:
                raw_text = await f.read()
                new_chapter = Chapter(
                    idx=idx,
                    raw_text=raw_text,
                    file_name=chapter,
                    script=None,
                )
                self.chapters.append(new_chapter)
        logger.info(f"Read {len(self.chapters)} chapters")

        # turn all chapters into scripts
        tsks = [
            asyncio.create_task(self.get_script(chapter)) for chapter in self.chapters
        ]
        await asyncio.wait(tsks)
        logger.info(f"Generated {len(self.chapters)} scripts")

        await self.get_voice_prompts(self.chapters)
        logger.info(f"Generated {len(self.character_traits)} voice prompts")

        json.dump(
            self.character_traits,
            open("./data/xjxz/character_traits.json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4,
        )
        json.dump(
            [chapter.script for chapter in self.chapters],
            open("./data/xjxz/chapters.json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4,
        )

    async def get_script(self, chapter: Chapter):
        result = await self.providers["llm"].generate(
            prompt=GET_SPEAKER_PROMPT.format(
                text=chapter.raw_text,
            ),
            schema={
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "array",
                        "properties": {
                            "speaker": {"type": "string"},
                            "content": {"type": "string"},
                            "emotion": {"type": "string"},
                        },
                    }
                },
            },
        )
        # logger.info(result)
        chapter.script = json.loads(result)
        return 0

    async def get_voice_prompts(self, chapters: list[Chapter]):
        for chapter in tqdm(chapters):
            characters = list(set(speech["speaker"] for speech in chapter.script))
            characters = list(
                filter(
                    lambda x: x != "旁白" and x not in self.character_traits.keys(),
                    characters,
                )
            )

            result = await self.providers["trait_guesser"].guess(
                text=chapter.raw_text,
                characters=characters,
            )

            for trait in result:
                if trait["character"] in self.character_traits.keys():
                    logger.info(f"{trait['character']} already exists, skip")

                self.character_traits[trait["character"]] = trait["voice_prompt"]


if __name__ == "__main__":
    bookcaster = Bookcaster(
        path="./data/xjxz",
        providers={
            "llm": LLMProvider(
                model="qwen3.5-9b-uncensored-hauhaucs-aggressive",
                api_key="lmstudio",
                base_url="http://snowfox4004.local:1234/v1/",
            ),
        },
    )
    asyncio.run(bookcaster.podcast())
