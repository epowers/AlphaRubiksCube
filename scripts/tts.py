#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

try:
    import TTS as __TTS
except ImportError:
    print("ERROR: cannot find TTS... exiting.")
    sys.exit(1)
try:
    import ffmpeg as __ffmpeg
except ImportError:
    print("ERROR: cannot find ffmpeg... exiting.")
    sys.exit(1)


class TTS:
    def __init__(self):
        import TTS
        from TTS.utils.manage import ModelManager
        from TTS.utils.synthesizer import Synthesizer

        path = Path(TTS.__file__).parent / ".models.json"
        manager = ModelManager(path, progress_bar=False)

        model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        model_path, config_path, model_item = manager.download_model(model_name)
        vocoder_name = model_item["default_vocoder"]
        vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)

        speakers_file_path = None
        language_ids_file_path = None
        encoder_path = None
        encoder_config_path = None

        self.synthesizer = Synthesizer(
            model_path,
            config_path,
            speakers_file_path,
            language_ids_file_path,
            vocoder_path,
            vocoder_config_path,
            encoder_path,
            encoder_config_path,
            False,
        )

    def synthesize(self, text, out_path):
        wav = self.synthesizer.tts(
            text,
            None,
            None,
            None,
            reference_wav=None,
            style_wav=None,
            style_text=None,
            reference_speaker_name=None,
        )
        self.synthesizer.save_wav(wav, out_path)

    @classmethod
    def concat(self, inputs, output):
        import ffmpeg
        ffmpeg.concat(
            *(ffmpeg.input(in_) for in_ in inputs),
            a=1, v=0
        ).output(output).run(overwrite_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=Path)
    parser.add_argument('destination_folder', type=Path, default='.')
    args = parser.parse_args()

    data_dict = dict()
    with open(args.input_file, "r") as input_file:
        input_file.seek(0)
        filename = 'default'
        while True:
            line = input_file.readline()
            if line == '':
                break
            line = line.strip()
            if line == '':
                continue
            if line[0] == '#':
                continue
            if ':' in line:
                filename, _ = line.split(':', 1)
                filename = filename.strip()
                continue
            if filename not in data_dict:
                data_dict[filename] = []
            data_dict[filename].append(line)

    tts = TTS()
    for filename, lines in data_dict.items():
        path = Path(args.destination_folder) / filename
        n_lines = len(lines)
        inputs = [None] * n_lines
        for i, text in enumerate(lines):
            if n_lines > 1:
                out_path = inputs[i] = f'{path}_{i:02d}.wav'
            else:
                out_path = f'{path}.wav'
            tts.synthesize(text, out_path)
        if n_lines > 1:
            out_path = f'{path}.wav'
            TTS.concat(inputs, out_path)
            for in_ in inputs:
                os.remove(in_)


if __name__ == "__main__":
    main()
