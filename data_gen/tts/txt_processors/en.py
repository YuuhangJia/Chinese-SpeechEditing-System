# coding=gbk

import re
import unicodedata

from g2p_en import G2p
from g2p_en.expand import normalize_numbers
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer

from data_gen.tts.txt_processors.base_text_processor import BaseTxtProcessor, register_txt_processors
from utils.text.text_encoder import PUNCS, is_sil_phoneme


class EnG2p(G2p):
    word_tokenize = TweetTokenizer().tokenize

    def __call__(self, text):
        # preprocessing
        words = EnG2p.word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag) #　词性标注

        # steps
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None: #　re.search，查找word中是否有[a-z]
                pron = [word]

            elif word in self.homograph2features:  # Check homograph 同形异义词(拼写相同，意义不同，读音可能不同)
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                pron = self.cmu[word][0]
            else:  # predict for oov
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]


@register_txt_processors('en')
class TxtProcessor(BaseTxtProcessor):
    g2p = EnG2p()

    @staticmethod
    def preprocess_text(text): #　@classmethod　静态方法通常用于实现与类实例无关的功能
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub("[\'\"()]+", "", text) # 替换正则表达式为空;
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ a-z{PUNCS}]", "", text)
        text = re.sub(f" ?([{PUNCS}]) ?", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = text.replace("i.e.", "that is")
        text = text.replace("i.e.", "that is")
        text = text.replace("etc.", "etc")
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r" ", text)
        return text

    @classmethod
    def process(cls, txt):
        txt = cls.preprocess_text(txt).strip() #strip()方法可以删除字符串开头和结尾的空格
        phs = cls.g2p(txt)
        txt_struct = [[w, []] for w in txt.split(" ")]
        i_word = 0
        for p in phs:
            if p == ' ':
                i_word += 1
            else:
                txt_struct[i_word][1].append(p) #　txt_struct：［｛w, [ph]}, {w, [ph]}, ...］
        txt_struct = cls.postprocess(txt_struct)
        return txt_struct, txt
