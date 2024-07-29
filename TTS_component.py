'''
class: TTService
    inner interface:
        __read : 处理输入文本并生成音频数据块的生成器
        __change_text : 将输入文本转换为模型可处理的格式，包括文本规范化和张量转换
        __chunk_in_infer : 将长文本分块处理
    outer interface:
        text_to_audio : 将文本转换为音频文件
            input: text, audio_file_path
            output: text
'''
import sys
import time
import logging

sys.path.append('TTS/vits') # 将TTS/vits添加到系统路径中

import soundfile
import os
os.environ["PYTORCH_JIT"] = "0"
import torch

import TTS.vits.commons as commons
import TTS.vits.utils as utils
from TTS.vits.models import SynthesizerTrn
from TTS.vits.text.symbols import symbols
from TTS.vits.text import text_to_sequence

class TTService:
    def __init__(self,config,model_path,character,speed):
        logging.info("Initializing TTS Service...")
        start_time = time.time()
        self._speed=speed
        self._hyperparameters=utils.get_hparams_from_file(config)
        self._network_generate = SynthesizerTrn(
            len(symbols),
            self._hyperparameters.data.filter_length//2+1,
            self._hyperparameters.train.segment_size//self._hyperparameters.data.hop_length,
            **self._hyperparameters.model).cuda()
        _ = self._network_generate.eval()
        _ = utils.load_checkpoint(model_path, self._network_generate, None)
        self._network_generate.infer(self.__change_text("你好啊，我是旅行者。")[0],self.__change_text("你好啊，我是旅行者。")[1],noise_scale=0.667,length_scale=self._speed)

        logging.info('Loading VITS model @Time used: %.2f seconds', time.time() - start_time)

    def __chunk_in_infer(self,x_text,x_text_lengths,chunk_size=256): # chunk_size 控制了每次处理的文本块大小 best: 256
        length = x_text_lengths.item()
        for start in range(0, length, chunk_size):
            end = min(length, start + chunk_size)
            x_text_chunk = x_text[:,start:end]
            x_text_chunk_length = torch.LongTensor([end-start]).cuda()
            audio_chunk=self._network_generate.infer(x_text_chunk,x_text_chunk_length,noise_scale=0.667,length_scale=self._speed)[0][0,0].data.cpu().float().numpy()
            yield audio_chunk

    def __change_text(self,text):
        text_norm = text_to_sequence(text, self._hyperparameters.data.text_cleaners)
        if self._hyperparameters.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        with torch.no_grad():
            x_text = text_norm.cuda().unsqueeze(0)
            x_text_lengths = torch.LongTensor([text_norm.size(0)]).cuda()
            return x_text,x_text_lengths

    def __read(self,text):
            text=text.replace('~','！')
            x_text,x_text_lengths = self.__change_text(text)
            for chunk in self.__chunk_in_infer(x_text,x_text_lengths):
                yield chunk

    def text_to_audio(self, text, file_path):
        start_time = time.time()
        with soundfile.SoundFile(file_path, 'w', samplerate = self._hyperparameters.data.sampling_rate,channels=1) as f:
            for chunk in self.__read(text):
                f.write(chunk)
        logging.info("Text To Audio @Time used: %.2f", time.time()-start_time)


if __name__ == "__main__":
    config_combo = [
        # ("TTS/models/CyberYunfei3k.json", "TTS/models/yunfei3k_69k.pth"),
        ("TTS/models/paimon6k.json", "TTS/models/paimon6k_390k.pth"),
        # ("TTS/models/ayaka.json", "TTS/models/ayaka_167k.pth"),
        # ("TTS/models/ningguang.json", "TTS/models/ningguang_179k.pth"),
        # ("TTS/models/nahida.json", "TTS/models/nahida_129k.pth"),
        # ("TTS/models_unused/miko.json", "TTS/models_unused/miko_139k.pth"),
        # ("TTS/models_unused/yoimiya.json", "TTS/models_unused/yoimiya_102k.pth"),
        # ("TTS/models/noelle.json", "TTS/models/noelle_337k.pth"),
        # ("TTS/models_unused/yunfeimix.json", "TTS/models_unused/yunfeimix_122k.pth"),
        # ("TTS/models_unused/yunfeineo.json", "TTS/models_unused/yunfeineo_25k.pth"),
        # ("TTS/models/yunfeimix2.json", "TTS/models/yunfeimix2_47k.pth")
        # ("TTS/models_unused/zhongli.json", "TTS/models_unused/zhongli_44k.pth"),
    ]
    tts = TTService(config_combo[0][0], config_combo[0][1], 'test', 1)
    text="旅行者，今天是星期四，能否威我八十,让我去嗨皮一下！八十可能不够，那就威我一百！锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"
    file_path="./output.wav"
    tts.text_to_audio(text, file_path)
