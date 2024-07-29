'''
class: ASRService
    inner interface(内部属性和方法):audio->text
    outer interface:
        input: audio_file_path
        output: text
'''
import logging
import time
from pathlib import Path
from ASR.rapid_paraformer import RapidParaformer

class ASRService:
    def __init__(self):
        logging.info('Initializing ASR Service...')

        self._config_path = Path(__file__).parent / 'resources' / 'config.yaml'
        start_time = time.time()
        self.paraformer = RapidParaformer(self._config_path)

        logging.info('Loading ASR model @Time used: %.2f seconds', time.time() - start_time)


    def audio_to_text(self, wav_path):
        start_time = time.time()
        result = self.paraformer(wav_path)
        logging.info('ASR Result: %s @Time used %.2f.' % (result[0], time.time() - start_time))
        return result[0]


# test : given audio, return text
if __name__ == '__main__':

    service = ASRService()

    wav_path = "./test_wavs/test_long_story.wav"

    result = service.audio_to_text(wav_path)
    if result:
        print(result)
    else:
        print('Failed to process audio.')
