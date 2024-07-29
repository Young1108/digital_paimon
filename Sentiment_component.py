'''
class SentimentService:
    inner:  __infer(text)
    outer:
        input: text (From LLM)
        output: sentiment_tag

'''
import logging
import onnxruntime
from transformers import BertTokenizer
import numpy as np
import time

class SentimentService():
    def __init__(self):
        logging.info('Initializing Sentiment Service...')
        self._model_path = 'Sentiment/models/paimon_sentiment.onnx'
        start_time = time.time()
        self.ort_session = onnxruntime.InferenceSession(self._model_path, providers=['CPUExecutionProvider'])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        logging.info('Loading BERT model @Time used: %.2f seconds', time.time() - start_time)

    def __infer(self, text):
        tokens = self.tokenizer(text, return_tensors="np")
        input_dict = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
        input_dict["input_ids"] = input_dict["input_ids"].astype(np.int64)
        input_dict["attention_mask"] = input_dict["attention_mask"].astype(np.int64)
        logits = self.ort_session.run(["logits"], input_dict)[0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        label = np.argmax(probabilities, axis=1)[0]
        logging.info(f'Sentiment Engine Infer: {label}')
        return label

    def get_sentiment_tag(self, text):
        start_time=time.time()
        sentiment_tag = self.__infer(text)
        logging.info(f"Sentiment Time used: {time.time()-start_time}")
        return sentiment_tag


if __name__ == '__main__':
    test_text = '不许你这样说我，打你'
    s = SentimentService()
    res = s.get_sentiment_tag(test_text)
    print(res)
