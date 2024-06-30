import asyncio
import logging
import math
import io

from styletts2 import tts
from typing import Union
from pydub import AudioSegment
import soundfile as sf

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.core.utils import set_audio_tags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider

logger = logging.getLogger(__name__)



# Credit: https://gist.github.com/moha-abdi/8ddbcb206c38f592c65ada1e5479f2bf
# @phuchoang2603 contributed pause support in https://github.com/p0n1/epub_to_audiobook/pull/45
class CommWithPauses:
    # This class uses Style TTS 2 to generate text
    # but with pauses for example:- text: 'Hello
    # this is simple text. [pause: 1000] Paused 1000ms'
    def __init__(
        self,
        text: str,
        break_string: str,
        break_duration: int = 1250,
        tts_model: tts.StyleTTS2 = None,
        **kwargs,
    ) -> None:
        self.full_text = text
        self.break_string = break_string
        self.break_duration = int(break_duration)

        if tts_model is None:
            raise ValueError("No TTS model provided")
        
        self.tts_model = tts_model

        self.parsed = self.parse_text()
        self.file = io.BytesIO()

    def parse_text(self):
        logger.debug(
            f"Parsing the text, looking for break/pauses in text: <{self.full_text}>"
        )
        if self.break_string not in self.full_text:
            logger.debug(f"No break/pauses found in the text")
            return [self.full_text]

        parts = self.full_text.split(self.break_string)
        logger.debug(f"split into <{len(parts)}> parts: {parts}")
        return parts

    async def chunkify(self):
        logger.debug(f"Chunkifying the text")
        for content in self.parsed:
            logger.debug(f"content from parsed: <{content}>")
            audio_bytes = await self.generate_audio(content)
            self.file.write(audio_bytes)
            if content != self.parsed[-1] and self.break_duration > 0:
                # only same break duration for all breaks is supported now
                pause_bytes = self.generate_pause(self.break_duration)
                self.file.write(pause_bytes)
        logger.debug(f"Chunkifying done")

    def generate_pause(self, time: int) -> bytes:
        logger.debug(f"Generating pause")
        # pause time should be provided in ms
        silent: AudioSegment = AudioSegment.silent(time, 24000)
        return silent.raw_data  # type: ignore

    async def generate_audio(self, text: str) -> bytes:
        logger.debug(f"Generating audio for: <{text}>")

        # Generate audio data as a NumPy array from the TTS model
        audio_data = self.tts_model.inference(text=text , output_sample_rate=24000,
                                              diffusion_steps=20)  
        logger.debug("Generated audio data as a NumPy array")

        # Convert the NumPy array to a WAV format in memory
        temp_wav = io.BytesIO()
        sf.write(temp_wav, audio_data, 24000, format='WAV')
        temp_wav.seek(0)
        
        # Load the WAV data into pydub's AudioSegment
        try:
            logger.debug("Decoding the WAV data")
            audio_segment = AudioSegment.from_file(temp_wav, format='wav')
        except Exception as e:
            logger.warning(f"Failed to decode the WAV data, reason: {e}, returning a silent chunk.")
            audio_segment = AudioSegment.silent(duration=1000, frame_rate=24000)
        
        logger.debug("Returning the decoded audio segment raw data")

        return audio_segment.raw_data

    async def save(
        self,
        audio_fname: Union[str, bytes],
    ) -> None:
        await self.chunkify()

        self.file.seek(0)
        audio: AudioSegment = AudioSegment.from_raw(
            self.file, sample_width=2, frame_rate=24000, channels=1
        )
        logger.debug(f"Exporting the audio")
        audio.export(audio_fname)
        logger.info(f"Saved the audio to: {audio_fname}")


class StyleTTS2sProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        logger.setLevel(config.log)
        # TTS provider specific config
        config.output_format = config.output_format or "audio-24khz-48kbitrate-mono-mp3"
        config.proxy = config.proxy or None

        # Have params to the the voice if wanted for now leave blank

        # 0.000$ per 1 million characters
        # or 0.000$ per 1000 characters
        self.price = 0.000
        
        # Initilise the TTS model 
        self.style_tts = self.initlise_style_tts2()

        super().__init__(config)
        


    def initlise_style_tts2(self):
        my_tts = tts.StyleTTS2()
        return my_tts

    def __str__(self) -> str:
        return f"{self.config}"

    def validate_config(self):
        # No validation needed for now
        return None

    def text_to_speech(
        self,
        text: str,
        output_file: str,
        audio_tags: AudioTags,
    ):

        communicate = CommWithPauses(
            text=text,
            break_string=self.get_break_string().strip(),
            break_duration=int(self.config.break_duration),
            tts_model = self.style_tts
        )

        asyncio.run(communicate.save(output_file))

        set_audio_tags(output_file, audio_tags)

    def estimate_cost(self, total_chars):
        return math.ceil(total_chars / 1000) * self.price

    def get_break_string(self):
        return " @BRK#"

    def get_output_file_extension(self):
        if self.config.output_format.endswith("mp3"):
            return "mp3"
        else:
            # Only mp3 supported in edge-tts https://github.com/rany2/edge-tts/issues/179
            raise NotImplementedError(
                f"Unknown file extension for output format: {self.config.output_format}. Only mp3 supported in edge-tts. See https://github.com/rany2/edge-tts/issues/179."
            )
