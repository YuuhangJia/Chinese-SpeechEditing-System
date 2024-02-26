import re # 正则化处理有关
import jieba # 结巴分词
from pypinyin import pinyin, Style # 中文转拼音
from data_gen.tts.text_normalize import NSWNormalizer # 中文的正则化：例如189: 幺八九; 注，text_normalize.py需用utf-8进行保存;

class ChineseTxtProcessor():
    @classmethod
    def full_angle_to_half(cls, text):
        table = {ord(f): ord(t) for f, t in zip(
            u'：，。！？【】（）％＃＠＆１２３４５６７８９０',
            u':,.!?[]()%#@&1234567890')}  # 全角转半角，字典{全角：半角}
        
        return text.translate(table) # str类型的内置方法translate，传入一个table即可完成翻译；
    
    @classmethod
    def regular_modify(cls, text):
        PUNCS = '!,.?;:'
        text = re.sub("[\'\"()]+", "", text) # 去掉单双引号
        text = re.sub("[-]+", " ", text) # 去掉横杆
        text = re.sub(f"[^ A-Za-z\u4e00-\u9fff{PUNCS}]", "", text)
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r"", text) 
        text = re.sub(rf"[A-Za-z]+", r"$", text)  # 将英文单词替换为$
        return text
    
    @classmethod
    def process_with_en(cls, pinyin): # 处理汉语中夹杂的英文, 同时将二维列表转为一维;
        pinyin = [item[0] if '$' not in item else 'ENG' for item in pinyin] #将所有的英文('$')全部替换为'ENG'
        return pinyin
    
    @classmethod
    def text_to_phonme(cls, text, use_tone=True):
        shengmu = pinyin(text, style=Style.INITIALS, strict=False)
        if use_tone:
            yunmu = pinyin(text, style= Style.FINALS_TONE3,strict=False)
        else:
            yunmu = pinyin(text, style= Style.FINALS,strict=False) # Style.FINALS是没有声调的选项
            
        # print("shengmu: ", shengmu)   # shengmu:  [['$'], ['y'], ['d'], ...
        # print("yunmu: ", yunmu)     # yunmu:  [['$'], ['u4'], ['ao4'], ...
        
        shengmu = cls.process_with_en(shengmu)
        yunmu = cls.process_with_en(yunmu)
        
        assert len(shengmu) == len(yunmu)
        ph_list = []
        for s, y in zip(shengmu, yunmu):
            if s == y:
                ph_list += [s]
            else:
                ph_list += [s + "%" + y]
        # print(ph_list) # ['ENG', 'y%u4', 'd%ao4', 'w%ei1', 'x%ian3', ...
        return ph_list
    
    @classmethod
    def segmentation(cls, text, ph_list):
        seg_text = '#'.join(jieba.cut(text))
        # print(seg_text)  # $#遇到#危险#后#,#在#第一#时间#拨打#了#一百一十#,#并#喊道#救命#.
        
        assert len(ph_list) == len([s for s in seg_text if s != '#'])
        seg_ph_list = []
        idx = 0
        seg_flag = False
        for txt in seg_text:
            if txt == '#':
                seg_ph_list.append('#')
                seg_flag = True
            else:
                if len(seg_ph_list) > 0 and not seg_flag:
                    seg_ph_list.append("|") 
                seg_ph_list += (i for i in ph_list[idx].split("%") if i != '')
                idx += 1
                seg_flag = False
                
        # print(seg_ph_list)  # ['ENG', '#', '|', 'y', 'u4', '|', 'd', 'ao4', '#', '|', 'w', 'ei1', '|', 'x', 'ian3', '#', '|', 'h', 'ou4', '#',  ... 其中#分词，|分字;
        return seg_ph_list
    
    @staticmethod
    def sp_phonemes():
        return ['|', '#']
    
    @classmethod
    def remove_sil_seg(cls, ph_list):
        PUNCS = '!,.?;:'
        sil_phonemes = list(PUNCS) + ChineseTxtProcessor.sp_phonemes()
        # print("ph_list: ", ph_list)
        ph_list_removal = []
        for i in range(0, len(ph_list), 1):
            if ph_list[i] != '#' or (ph_list[i - 1] not in sil_phonemes and ph_list[i + 1] not in sil_phonemes):
                ph_list_removal.append(ph_list[i])
        # print("ph_list_removal:", ph_list_removal) # 例如逗号周围的#号：[..., '#', ',', '#', ...]
        
        return ph_list_removal
    
    @classmethod
    def process(cls, text, use_tone=True):
        text = cls.full_angle_to_half(text) #全角转为半角, 解决编码问题;
        text = NSWNormalizer(text).normalize(remove_punc=False).lower() # 中文有关的处理，电话号码、数字处理等;
        text = cls.regular_modify(text) # 正则化处理;
        
        # 此时text便符合了pypinyin的输入要求。
        ph_list = cls.text_to_phonme(text, use_tone) # text => phonme
        ph_list = cls.segmentation(text, ph_list)
        ph_list = cls.remove_sil_seg(ph_list) # 去除静音符号周围的词边界标记
        
        return ph_list, text
