import os
import shutil

class GeneralConfig:
    def __init__(self, args):
        # Sanitize and process input file
        self.input_file = self.process_input_file(args.input_file)
        
        # General arguments
        self.output_folder = args.output_folder
        self.preview = args.preview
        self.output_text = args.output_text
        self.log = args.log
        self.no_prompt = args.no_prompt
        self.title_mode = args.title_mode

        # Book parser specific arguments
        self.newline_mode = args.newline_mode
        self.chapter_start = args.chapter_start
        self.chapter_end = args.chapter_end
        self.remove_endnotes = args.remove_endnotes

        # TTS provider: common arguments
        self.tts = args.tts
        self.language = args.language
        self.voice_name = args.voice_name
        self.output_format = args.output_format
        self.model_name = args.model_name

        # TTS provider: Azure & Edge TTS specific arguments
        self.break_duration = args.break_duration

        # TTS provider: Edge specific arguments
        self.voice_rate = args.voice_rate
        self.voice_volume = args.voice_volume
        self.voice_pitch = args.voice_pitch
        self.proxy = args.proxy

        # TTS provider: StyleTTS2 specific arguments
        self.diffusion_steps = args.diffusion_steps

    def __str__(self):
        return ', '.join(f"{key}={value}" for key, value in self.__dict__.items())
    
    # Added code to renmae input file if it contains invalid characters
    @staticmethod
    def sanitize_filename(filename):
        # Remove any characters that aren't alphanumeric, underscore, hyphen, or period
        return ''.join(c for c in filename if c.isalnum() or c in ['_', '-', '.'])

    def process_input_file(self, input_file):
        input_dir, input_filename = os.path.split(input_file)
        sanitized_filename = self.sanitize_filename(input_filename)
        
        if sanitized_filename != input_filename:
            new_input_file = os.path.join(input_dir, sanitized_filename)
            shutil.copy2(input_file, new_input_file)
            print(f"Input file renamed to: {sanitized_filename}")
            return new_input_file
        
        return input_file