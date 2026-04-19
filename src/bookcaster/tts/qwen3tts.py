import asyncio
import base64
import os

import httpx
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

from bookcaster.tts.tts_utils import transcode_mp3, concat_wav_bytes
from bookcaster.utils import Chapter

# ===== 变量设置 =====
_api_key = "empty"  # 留空从环境变量读取
_api_base = "http://localhost:8000/v1"  # 留空使用官方 API


class Qwen3TTS:
    def __init__(self, api_key: str = "Nothing", api_base: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    async def tts(self, chapters: list[Chapter], traits: dict | None = None):
        if traits is None:
            logger.warning("未设置角色特征，将使用纯随机音色")
            traits = {}

        for chapter in tqdm(chapters):

            script = chapter.script
            characters = set(speech["speaker"] for speech in script)
            character_speech = {character: [] for character in characters}

            for speech in script:
                character_speech[speech["speaker"]].append(speech)

            tsks = []
            character_list = list(character_speech.keys())
            for character in character_list:
                instruction = traits.get(character, "")
                if character == "旁白":
                    instruction = "沉稳厚重的男声，语速缓慢，语调低沉稳定，有抑扬顿挫。娓娓道来的语气。"

                task = asyncio.create_task(
                    self.tts_batch(
                        texts=[
                            speech["content"] for speech in character_speech[character]
                        ],
                        instruction=instruction,
                        response_format="wav",
                    )
                )
                tsks.append(task)

            tts_results = await asyncio.gather(*tsks)
            logger.info(
                str(sum(len(tts_result) for tts_result in tts_results))
                + " audio was generated",
            )
            pointers = [0] * len(character_speech)
            audios = []

            for speech in chapter.script:
                character = speech["speaker"]
                character_id = character_list.index(character)

                wav_byte = tts_results[character_id][pointers[character_id]]
                pointers[character_id] += 1

                audios.append(wav_byte)

            audio = concat_wav_bytes(audios)
            audio = transcode_mp3(audio)

            logger.info(f"concatenated {len(chapter.script)} audios, {len(audio)=}")
            chapter.audio = transcode_mp3(audio)

    async def tts_single(
        self,
        text: str,
        instruction: str,
        model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        voice: str = "vivian",
    ):
        """
        生成语音并保存为文件

        参数:
            text: 要转换为语音的文本
            output_path: 输出文件路径
            model: 模型名称（tts-1, tts-1-hd, gpt-4o-mini-tts）
            voice: 语音风格（alloy, echo, fable, onyx, nova, shimmer）
        """

        try:
            response = await self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                instructions=instruction,
                extra_body={"task_type": "VoiceDesign"},
            )
            wav_bytes = response.content
            mp3_bytes = transcode_mp3(wav_bytes)

            logger.info(f"语音已生成:{text[:80]}")
            return mp3_bytes
        except Exception as e:
            logger.error(f"生成语音时出错: {e}")
            return bytes()

    async def tts_batch(
        self,
        texts: list[str],
        instruction: str,
        response_format: str = "mp3",
        is_retrying: bool = False,
    ):

        items = [{"input": text} for text in texts]
        results = []

        for i in range(0, len(items), 32):
            payload = {
                "items": items[i : i + 32],
                "response_format": response_format,
                "voice": "vivian",
                "task_type": "VoiceDesign",
                "instructions": instruction,
            }

            headers = {
                "Authorization": f"Bearer {_api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    url=f"{_api_base}/audio/speech/batch", json=payload, headers=headers
                )

            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                return

            data = response.json()
            if data["succeeded"] < data["total"]:
                logger.warning(f"生成语音失败: {data['failed']} / {data['total']}")

                filtered_data = data
                for result in filtered_data["results"]:
                    result.pop("audio_data")

                logger.warning(filtered_data)

                # retry once
                if not is_retrying:
                    return await self.tts_batch(
                        texts, instruction, response_format, is_retrying=True
                    )

            for result in data["results"]:
                idx = result["index"]
                if result["status"] == "success":
                    audio_bytes = base64.b64decode(result["audio_data"])
                    results.append(audio_bytes)
                else:
                    logger.warning(f"{idx} has failed")

        return results


def generate_speech(
    text: str,
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    voice: str = "vivian",
):
    """
    生成语音并保存为文件

    参数:
        text: 要转换为语音的文本
        output_path: 输出文件路径
        model: 模型名称（tts-1, tts-1-hd, gpt-4o-mini-tts）
        voice: 语音风格（alloy, echo, fable, onyx, nova, shimmer）
    """
    client = OpenAI(
        api_key=_api_key if _api_key else os.environ.get("OPENAI_API_KEY"),
        base_url=_api_base if _api_base else "https://api.openai.com/v1",
    )
    try:
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            instructions="苍老的男性声音，急促地",
            extra_body={"task_type": "VoiceDesign"},
        )
        wav_bytes = response.content
        mp3_bytes = transcode_mp3(wav_bytes)

        logger.info(f"语音已生成:{text[:80]}")
        return mp3_bytes
    except Exception as e:
        logger.error(f"生成语音时出错: {e}")
        return bytes()


async def test():
    qwen = Qwen3TTS()

    texts = [
        "总结与选型建议",
        "要选择哪种方法，主要取决于你的具体需求：",
        "追求最高性能和精细控制：选择 PyAV (Python)、FFmpeg.Autogen (.NET) 等 底层绑定库。",
        "追求开发效率和简洁代码：选择 ffmpeg-python (Python)、FFMpegCore (.NET) 等封装型库。",
        "跨平台、无 环境依赖：考虑使用远程 API 服务。",
        "需要执行简单的命令行任务：继续使用 subprocess 等模块直接调用命令行本身也是一个完全可行的方案。",
    ]

    mp3 = await qwen.tts_batch(
        texts,
        "女声，快速地，声音活泼开朗",
    )
    output_dir = "data/tts_results/qwen3tts"
    os.makedirs(output_dir, exist_ok=True)

    for i, mp3 in enumerate(mp3):
        output_path = os.path.join(output_dir, f"{i}.mp3")
        with open(output_path, "wb") as f:
            f.write(mp3)


if __name__ == "__main__":
    # mp3 = generate_speech("这是一段测试文本。", "test_output.mp3")
    # open("./test_output.mp3", "wb").write(mp3)
    import asyncio

    asyncio.run(test())
{
    "items": [
        {"input": " 总结与选型建议"},
        {"input": "\n要选择哪种方法，主要取决于你的具体需求："},
        {
            "input": "追求最高性能和精细控制：选择 PyAV (Python)、FFmpeg.Autogen (.NET) 等底层绑定库。"
        },
        {
            "input": "追求开发效率和简洁代码：选择 ffmpeg-python (Python)、FFMpegCore (.NET) 等封装型库。"
        },
        {"input": "跨平台、无环境依赖：考虑使用远程 API 服务。"},
        {
            "input": "需要执行简单的命令行任务：继续使用 subprocess 等模块直接调用命令行本身也是一个完全可行的方案。"
        },
    ],
    "response_format": "mp3",
    "voice": "vivian",
    "task_type": "VoiceDesign",
    "instructions": "苍老的男性声音，急促地",
}
