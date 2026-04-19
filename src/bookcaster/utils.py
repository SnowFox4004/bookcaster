from dataclasses import dataclass


@dataclass
class Chapter:
    idx: int
    file_name: str
    raw_text: str
    script: list[dict[str, str]]
    audio: bytes | None = None
