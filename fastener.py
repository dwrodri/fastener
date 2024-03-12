import json
import logging
import os
import pathlib
import sys
import tempfile
import subprocess
from typing import Dict, List

import numpy as np

import whisper

logging.basicConfig(level=logging.INFO)


def check_for_ffmpeg_binaries() -> bool:
    """
    Check that ffmpeg and ffprobe are on path
    """
    # Get the PATH environment variable
    path_env_var = os.environ.get("PATH")

    if path_env_var is None:
        # If PATH is not set, just exit
        logging.critical(
            f"Obtained null environment path {path_env_var}, this script needs ffmpeg on your path!"
        )
        return False

    # search the paths for ffmpeg
    found_ffmpeg = False
    found_ffprobe = False
    for binary_folder in map(pathlib.Path, path_env_var.split(os.pathsep)):
        if binary_folder.exists():
            for binary in binary_folder.iterdir():
                found_ffmpeg = found_ffmpeg or binary.name == "ffmpeg"
                found_ffprobe = found_ffprobe or binary.name == "ffprobe"
                if found_ffmpeg and found_ffprobe:
                    break
    return found_ffprobe and found_ffmpeg


def get_stream_metadata(media_file: pathlib.Path) -> List[Dict]:
    """
    call ffprobe to get audio/video/subtitle stream addresses
    """
    ffprobe_command: str = (
        f"ffprobe -v quiet -print_format json -show_streams {str(media_file)}"
    )
    cmd_result = subprocess.run(
        ffprobe_command.split(),
        stdin=None,
        capture_output=True,
        encoding=sys.getdefaultencoding(),
        check=True,
    )
    metadata_bundles = [
        stream_dict
        for stream_dict in json.loads(cmd_result.stdout)["streams"]
        if stream_dict["codec_type"] in ("video", "audio", "subtitle")
    ]
    # Extra check to assert we have usable audio
    audio_stream_count = len(
        [bundle for bundle in metadata_bundles if bundle["codec_type"] == "audio"]
    )
    if audio_stream_count == 0:
        logging.critical("File has no audio to transcribe, exiting")
        sys.exit(-1)
    return metadata_bundles


# TODO: fix type annotation for destination_fp
def extract_audio_to_ndarray(
    media_file: pathlib.Path,
    stream_index: int,
) -> np.ndarray:
    """
    Extract audio from media file to 16KHz pulsecodes and return as flattened ndarray
    """
    ffmpeg_extract_command = f"ffmpeg -nostdin -i {media_file} -map 0:{stream_index} -c:a pcm_16le -ar 160000 -ac 1 -f s16le -"
    cmd_result = subprocess.run(
        ffmpeg_extract_command.split(), check=True, capture_output=True
    )
    return (
        np.frombuffer(cmd_result.stdout, dtype=np.int16).flatten().astype(np.float32)
        / 32678.0
    )


def main(media_file: pathlib.Path):
    stream_metadata_bundles = get_stream_metadata(media_file)
    audio_metadata_bundles = [
        bundle for bundle in stream_metadata_bundles if bundle["codec_type"] == "audio"
    ]
    if len(audio_metadata_bundles) > 1:
        logging.info(
            "Multiple audio streams found, will only process non-english audio streams"
        )
    # Prep the model
    model = whisper.load_model("large-v3")
    # Transcribe non-engish audio streams
    for bundle in audio_metadata_bundles:
        audio = extract_audio_to_ndarray(media_file, bundle["index"])
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.pad_or_trim(audio)
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        # detect the spoken language
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")  # type: ignore


if __name__ == "__main__":
    # TODO: Consider replacing w/ typer or argparse?
    media_file = pathlib.Path()
    # if len(sys.argv) != 3 or any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    #     print("Usage: python3 fastener.py [path to video file]")
    #     sys.exit(-1)
    media_file = pathlib.Path(sys.argv[1])
    # quick sanity check before entering main
    if media_file.exists() and media_file.is_file() and check_for_ffmpeg_binaries():
        main(media_file.absolute())
