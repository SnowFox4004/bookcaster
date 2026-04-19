import asyncio
import json
import os
from dataclasses import dataclass

import aiofiles
from loguru import logger
from tqdm import tqdm

from bookcaster.llms import LLMProvider
from bookcaster.prompts import *
from bookcaster.trait_guesser import AgeGenderGuesser
from bookcaster.tts.qwen3tts import Qwen3TTS
from bookcaster.utils import Chapter
from typing import Any

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

        if self.providers.get("qwen3tts", None) is None:
            self.providers["tts"] = Qwen3TTS(
                api_key="nothingbeatsajet2holiday",
                api_base="http://127.0.0.1:8000/v1",
            )

    def __init__(
        self,
        path: str,
        providers: dict[str, Any] | None = None,
    ):

        self.path = path
        self.chapters = []
        self.providers = providers if providers is not None else {}
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
                    script=list(),
                )
                self.chapters.append(new_chapter)
        logger.info(f"Read {len(self.chapters)} chapters")

        # turn all chapters into scripts
        tsks = [
            asyncio.create_task(self.get_script(chapter)) for chapter in self.chapters
        ]
        await asyncio.wait(tsks)
        logger.info(f"Generated {len(self.chapters)} scripts")

        json.dump(
            [chapter.script for chapter in self.chapters],
            open("./data/xjxz/chapters.json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4,
        )

        await self.get_voice_prompts(self.chapters)
        logger.info(f"Generated {len(self.character_traits)} voice prompts")

        json.dump(
            self.character_traits,
            open("./data/xjxz/character_traits.json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4,
        )

        await self.providers["tts"].tts(
            await self.get_tts_chapters(), self.character_traits
        )

        await self.save_results()
        logger.success("Podcast finished")

    async def get_tts_chapters(self):
        chapters = filter(
            lambda x: not os.path.exists(
                os.path.join(self.path, "audio", x.file_name + ".mp3")
            ),
            self.chapters,
        )
        return list(chapters)

    async def save_results(self):
        os.makedirs(os.path.join(self.path, "audio"), exist_ok=True)
        for chapter in self.chapters:
            async with aiofiles.open(
                os.path.join(self.path, "audio", chapter.file_name + ".mp3"), "wb"
            ) as f:
                await f.write(chapter.audio)

    async def get_script(self, chapter: Chapter):
        os.makedirs(os.path.join(self.path, "scripts"), exist_ok=True)
        script_file_name = os.path.join(
            self.path, "scripts", chapter.file_name + ".json"
        )
        if os.path.exists(script_file_name):
            logger.info(f"{chapter.file_name}'s script already exists, skip")
            chapter.script = json.load(open(script_file_name, "r", encoding="utf-8"))
            return 0

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

        chapter.script = list(filter(lambda x: x.get("content"), chapter.script))

        chapter.script = self.concatenate_same_speaker_speech(chapter.script)
        json.dump(
            chapter.script,
            open(script_file_name, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4,
        )
        return 0

    def concatenate_same_speaker_speech(self, script: list[dict[str, str]]):
        new_script = list()
        for i in range(len(script)):
            if i == 0:
                new_script.append(script[i])
                continue

            if script[i]["speaker"] == script[i - 1]["speaker"]:
                new_script[-1]["content"] += " " + script[i]["content"]
            else:
                new_script.append(script[i])
        return new_script

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
                model="qwen3.5-4b",
                api_key="lmstudio",
                base_url="http://snowfox4004.local:1234/v1/",
            ),
        },
    )
    asyncio.run(bookcaster.podcast())
