import io
import logging
import math
from txtsplit import txtsplit


from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.utils import split_text, set_audio_tags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider


import style_tts2.styletts2importable as styletts2importable
import style_tts2.ljspeechimportable as ljspeechimportable
import torch
import os
import numpy as np
import pickle
from scipy.io import wavfile

logger = logging.getLogger(__name__)


def get_supported_models():
    #! For now adding support for the basic LJ Speech model
    return ["LJSpeech", "LJSpeech-Long","Voice_Cloning" , "Multi-Voice"]


def get_supported_voices():
    #? if Multi-Voice select the defult voice list can add more in the future
    return ['f-us-1', 'f-us-2', 'f-us-3', 'f-us-4', 'm-us-1', 'm-us-2', 'm-us-3', 'm-us-4']


def get_supported_formats():
    #! for now only support the wav format
    return ["wav"]


class StyleTTS2Local(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        logger.setLevel(config.log)
        config.model_name = config.model_name or "LJSpeech"
        config.voice_name = config.voice_name or "m-us-1"
        config.output_format = config.output_format or "wav"
        config.diffusion_steps = config.diffusion_steps or 10


        self.config = config    
    
        # Free as we are porcessing the model locally
        self.price = 0.000
        super().__init__(config)

    
    def __str__(self) -> str:
        return super().__str__()

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        audio_segments = []

        if self.config.model_name == "LJSpeech-Long":
            texts = txtsplit(text)
            s_prev = None # previus text adding to keep constantcy to the model
            for i, chunk in enumerate(texts , 1):
                logger.debug(
                    f"Processing chunk {i} of {len(texts)}, length={len(chunk)}, text=[{chunk}]"
                )

                logger.info(
                    f"Processing chapter-{audio_tags.idx} <{audio_tags.title}>, chunk {i} of {len(texts)}"
                )

                logger.debug(f"Text: [{chunk}], length={len(chunk)}")


                if text.strip() == "": continue
                text += '.' # add it back
                noise = torch.randn(1,1,256).to(device)
                audio , s_prev = ljspeechimportable.LFinferenceLong(chunk, s_prev, noise, 
                                                     diffusion_steps=self.config.diffusion_steps, 
                                                     embedding_scale=1)
                audio_segments.append(audio)
                logger.debug(f"Generated audio segment {i}")
        
        if self.config.model_name == "Multi-Voice":
            texts = txtsplit(text)
            for i, chunk in enumerate(texts, 1):
                noise = torch.randn(1,1,256).to(device)

                logger.debug(
                    f"Processing chunk {i} of {len(texts)}, length={len(chunk)}, text=[{chunk}]"
                )
                logger.info(
                    f"Processing chapter-{audio_tags.idx} <{audio_tags.title}>, chunk {i} of {len(texts)}"
                )

                logger.debug(f"Text: [{chunk}], length={len(chunk)}")
                
                set_voice = self.config.voice_name.lower()

                supported_voices = get_supported_voices()
                voices = {}

                # Load the list of voices
                for v in supported_voices:
                    voices[v] = styletts2importable.compute_style(f'style_tts2/voices/{v}.wav')
                    logger.debug(f"Loaded voice {v}")


                audio_segments.append(styletts2importable.inference(chunk, voices[set_voice], alpha=0.3, beta=0.7, 
                                                                    diffusion_steps=self.config.diffusion_steps, 
                                                                    embedding_scale=1)
)
                logger.debug(f"Generated audio segment {i}")
        else:
            texts = txtsplit(text)
            for i, chunk in enumerate(texts, 1):
                noise = torch.randn(1,1,256).to(device)

                logger.debug(
                    f"Processing chunk {i} of {len(texts)}, length={len(chunk)}, text=[{chunk}]"
                )
                logger.info(
                    f"Processing chapter-{audio_tags.idx} <{audio_tags.title}>, chunk {i} of {len(texts)}"
                )

                logger.debug(f"Text: [{chunk}], length={len(chunk)}")

                audio_segments.append(ljspeechimportable.inference(chunk, noise, 
                                                                diffusion_steps=self.config.diffusion_steps, 
                                                                embedding_scale=1))
                logger.debug(f"Generated audio segment {i}")

        # Combine all audio segments
        combined_audio = np.concatenate(audio_segments)

        sample_rate = 24000  
        wavfile.write(output_file, sample_rate, combined_audio)
        
        logger.info(f"Saved audio to {output_file}")
        
        set_audio_tags(output_file, audio_tags)

    def get_break_string(self):
        return "   "

    def get_output_file_extension(self):
        return self.config.output_format

    def validate_config(self):
        if self.config.model_name not in get_supported_models():
            raise ValueError(f"OpenAI: Unsupported model name: {self.config.model_name}")
        if self.config.voice_name not in get_supported_voices():
            raise ValueError(f"OpenAI: Unsupported voice name: {self.config.voice_name}")
        if self.config.output_format not in get_supported_formats():
            raise ValueError(f"OpenAI: Unsupported output format: {self.config.output_format}")

    def estimate_cost(self, total_chars):
        return math.ceil(total_chars / 1000) * self.price
