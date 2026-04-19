import io
from pydub import AudioSegment


def concat_wav_bytes(wav_bytes_list):
    """
    拼接多个 WAV 文件的 bytes 数据，返回合并后的 WAV bytes。
    """
    segments = [AudioSegment.from_wav(io.BytesIO(b)) for b in wav_bytes_list]
    combined = segments[0]
    for seg in segments[1:]:
        combined += seg
    # 导出为 WAV 格式的 bytes
    output = io.BytesIO()
    combined.export(output, format="wav")

    return output.getvalue()


def transcode_mp3(wav_bytes: bytes):
    import ffmpeg

    process = (
        ffmpeg.input("pipe:0")
        .output("pipe:1", format="mp3")
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )

    out_bytes, err_bytes = process.communicate(input=wav_bytes)
    if process.returncode != 0:
        raise RuntimeError(f"Error transcoding MP3: {err_bytes.decode('utf-8')}")

    return out_bytes
