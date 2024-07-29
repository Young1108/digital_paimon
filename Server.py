import argparse
import socket
import os
import librosa
import soundfile
import time
import logging

from ASR.ASR_component import ASRService
from TTS.TTS_component import TTService
from Sentiment.Sentiment_component import SentimentService
from LLM.LLM_component import LLMServeice, AccessTokenProvider
from utils.FlushingFileHandler import FlushingFileHandler

CHARACTER_NAME = {
            'paimon': ['TTS/models/paimon6k.json', 'TTS/models/paimon6k_390k.pth', 'character_paimon', 1],
            'yunfei': ['TTS/models/yunfeimix2.json', 'TTS/models/yunfeimix2_53k.pth', 'character_yunfei', 1.1],
            'catmaid': ['TTS/models/catmix.json', 'TTS/models/catmix_107k.pth', 'character_catmaid', 1.2]
        }


class Sever():
    def __init__(self,character:str):
        logging.info("Initializing server...")

        self._host = socket.gethostbyname(socket.gethostname())
        self._port=38438
        self._addr=None
        self._conn=None

        self._character=character
        self._tmp_recv_file_path = r'C:\workspace\new_digital_man\tmp\server_received.wav'
        self._tmp_processed_file_path = r'C:\workspace\new_digital_man\tmp\server_processed.wav'

        self._sever_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sever_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 10240000)
        self._sever_socket.bind((self._host, self._port))

        self._ASR=None
        self._LLM=None
        self._TTS=None
        self._BERT=None

        logging.info("Server initialized successfully!")

    def module_injection(self, asr:ASRService, bert:SentimentService, llm:LLMServeice, tts:TTService):
        try:
            start_time=time.time()
            logging.info("Module injecting...")
            self._ASR=asr
            self._BERT=bert
            self._LLM=llm
            self._TTS=tts
            logging.info("Module injected successfully!")
        except:
            raise TypeError("Module injection not implemented, please check the input parameters' type!")


    def listen(self):
        while True:
            self._sever_socket.listen()
            logging.info(f"Server is listening on {self._host}:{self._port}...")

            self._conn, self._addr = self._sever_socket.accept()
            logging.info(f"Connected by {self._addr}")

            self._conn.sendall(b"%s" % self._character.encode())
            while True:
                try:
                    receive_data = self.__receive_file()
                    with open(self._tmp_recv_file_path, 'wb') as f:
                        f.write(receive_data)
                        logging.info("Received file from %s", self._addr)
                    ask_text=self.__audio_to_text()
                    for answer in self._LLM.LLM_infer(ask_text):
                        self.__send_message(answer)
                    self.__notice_stream_end()

                except Exception as e:
                    print(e)

    def __notice_stream_end(self):
        time.sleep(0.5)
        self._conn.sendall(b'Send message finished')

    def __send_message(self, text):
        self._TTS.text_to_audio(text, self._tmp_processed_file_path)
        with open(self._tmp_processed_file_path, 'rb') as f:
            send_data = f.read()
        sentiment_tag = self._BERT.get_sentiment_tag(text)
        send_data+=b'?!'
        send_data+=b'%i'% sentiment_tag
        self._conn.sendall(send_data)
        time.sleep(0.5)
        logging.info("Sent message to %s, size %i", self._addr, len(send_data))

    def __processed_wav_file(self):
        with open(self._tmp_recv_file_path, 'r+b') as f:
            size_of_file = os.path.getsize(self._tmp_recv_file_path)-8
            f.seek(4)
            f.write(size_of_file.to_bytes(4, byteorder='little'))
            f.seek(40)
            f.write((size_of_file-28).to_bytes(4, byteorder='little'))
            f.flush()

    def __audio_to_text(self):
        self.__processed_wav_file()
        audio_signal,sample_rate=librosa.load(self._tmp_recv_file_path, sr=None, mono=False)
        mono_audio_signal = librosa.to_mono(audio_signal)
        mono_audio_signal = librosa.resample(mono_audio_signal, orig_sr=sample_rate, target_sr=16000)
        soundfile.write(self._tmp_recv_file_path,mono_audio_signal,16000)
        text = self._ASR.audio_to_text(self._tmp_recv_file_path)
        return text

    def __receive_file(self):
        receive_data=b''
        start_time = time.time()
        while True:
            time_gap=time.time()-start_time
            if time_gap>1:
                logging.info("Receive file timeout")
                break
            data = self._conn.recv(1024)
            self._conn.sendall(b'sb')
            if data[-2:]==b'?!':
                receive_data+=data[:-2]
                break
            if not data:
                logging.info("Wating for wav data...")
                continue
            else:
                receive_data+=data
                start_time = time.time()
        return receive_data

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--character", type=str, nargs='?', required=True)
    return parser.parse_args()

def custom_logging():
    console_logger = logging.getLogger()
    console_logger.setLevel(logging.INFO)
    format = '%(asctime)s %(levelname)s %(message)s'
    console_handler = console_logger.handlers[0]
    console_handler.setFormatter(logging.Formatter(format))
    console_logger.setLevel(logging.INFO)
    file_handler=FlushingFileHandler('log.log',formatter=logging.Formatter(format))
    file_handler.setLevel(logging.INFO)
    console_logger.addHandler(file_handler)
    console_logger.addHandler(console_handler)

def main(character_name):
    custom_logging()
    args=parse_args()
    try:
        asr=ASRService()
        bert=SentimentService()
        access_token_provider=AccessTokenProvider()
        llm=LLMServeice(access_token_provider)
        tts=TTService(*character_name[args.character])

        server=Sever(character_name[args.character][2])
        server.module_injection(asr, bert, llm, tts)
        server.listen()
    except Exception as e:
        print(e)

if __name__=="__main__":
    main(CHARACTER_NAME)

