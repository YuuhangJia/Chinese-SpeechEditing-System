import os
import re # 正则化处理有关
import json
import random
import itertools
from tqdm import tqdm

# from functools import partial
from collections import defaultdict, Counter
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.os_utils import link_file, move_file, remove_file
from utils.text.text_encoder import is_sil_phoneme, build_token_encoder
from data_gen.tts.zh import ChineseTxtProcessor

from typing import List
from dataclasses import dataclass

@dataclass
class ContentLine: 
    sid: str
    hans: List[str]
    pinyin: List[str]
    
def parse_content(path: str) -> List[ContentLine]: 
    res = []
    with open(path) as f: 
        for cols in (l.strip().split() for l in f): 
            sid, *content = cols
            hans = content[::2]
            pinyin = content[1::2]
            res.append(ContentLine(sid, hans, pinyin))
        return res 
    
import os.path as osp 

@dataclass
class MetaLine: 
    sid: str
    path: str
    spkid: str
    hans: str
    phones: str
    
def spk_from_sid(sid: str) -> str: 
    return sid[:7]
    
def content_to_meta(cl: ContentLine, join_path: str) -> MetaLine: 
    spkid = spk_from_sid(cl.sid)
    # phones, problems = get_phoneme_from_pinyin(cl.pinyin)
    # assert problems == [], f'in {cl}: {problems}'
    phones = cl.pinyin # 由于后续会使用mfa，所以使用pinyin作为音素；
    return MetaLine(
        cl.sid,
        osp.join(join_path, spkid, cl.sid), 
        spkid, 
        ''.join(cl.hans),
        ' '.join(phones)
    )

class BasePreprocessor:
    def __init__(self):
        self.dataset_name = "aishell3"
        self.txt_processor = ChineseTxtProcessor()
        
        self.raw_data_dir = f'data/raw/{self.dataset_name}'
        self.processed_dir = f'data/processed/{self.dataset_name}'
        self.temp_dir_for_process = f'data/temp_dir/{self.dataset_name}'
        
        self.mfa_input_dir = f'{self.processed_dir}/mfa_inputs'
        
        self.reset_phone_dict = True
        self.reset_word_dict = True
        self.word_dict_size = 12500
        self.num_spk = 218 # 218 for aishell3
        
        self.nsample_per_mfa_group = 2
        self.mfa_group_shuffle = False
        
    def meta_data(self):
        if self.dataset_name == "aishell3":
            base = self.raw_data_dir
            train_contents = parse_content(osp.join(base, 'train', 'content.txt'))
            test_contents  = parse_content(osp.join(base, 'test', 'content.txt'))
            
            metalines = [
                content_to_meta(tc, joinpath) for joinpath, tc in 
                tqdm(itertools.chain( # itertools.chain: Used to merge multiple iterables into a single iterable object
                    itertools.product([osp.join('train', 'wav')], train_contents), 
                    itertools.product([osp.join('test', 'wav')], test_contents)
                ), total=len(test_contents) + len(train_contents))
            ]
            
            # print("sid: ",metalines[0].sid) # sid:  SSB00050001.wav
            # print("path: ",metalines[0].path) # path:  train/wav/SSB0005/SSB00050001.wav
            # print("spkid: ",metalines[0].spkid) # spkid:  SSB0005
            # print("hans: ",metalines[0].hans) # hans:  广州女大学生登山失联四天警方找到疑似女尸
            # print("phones: ",metalines[0].phones) # phones:  guang3 zhou1 nv3 da4 xue2 sheng1 deng1 shan1 shi1 lian2 ...
            
            
            metaline_by_spkid = defaultdict(list)
            for ml in metalines: metaline_by_spkid[ml.spkid].append(ml)
            print(f'total speakers: {len(metaline_by_spkid)}')
            
            abundant_speakers = [spkid for spkid, xs in metaline_by_spkid.items() if len(xs) > 219]
            print(f'number of abundant speakers: {len(abundant_speakers)}')
            
            train_metas = []
            test_metas  = [] 
            for ml in metalines: 
                if ml.spkid in abundant_speakers: 
                    train_metas.append(ml)
                else: 
                    test_metas.append(ml)

            print(f'train lines: {len(train_metas)}')
            print(f'test lines: {len(test_metas)}')
            print(f'total lines: {len(train_metas) + len(test_metas)}')
            
            # 直接合并训练集和测试集：
            for meta in train_metas+test_metas:
                yield {'item_name': meta.sid, 'wav_fn': meta.path, 'txt': meta.hans, 'phones': meta.phones, 'spk_name': meta.spkid}
    
    
    def txt_to_ph(self, txt_raw):
        if self.dataset_name == "aishell3":
            phs, txt = self.txt_processor.process(txt_raw, {'use_tone': True})
            phs = [p.strip() for p in phs if p.strip() != ""]

            # 去除首尾的静音词
            while len(phs) > 0 and is_sil_phoneme(phs[0]):
                phs = phs[1:]
            while len(phs) > 0 and is_sil_phoneme(phs[-1]):
                phs = phs[:-1]
            phs = ["<BOS>"] + phs + ["<EOS>"] # 添上开始结束符
            phs_ = []
            for i in range(len(phs)):
                if len(phs_) == 0 or not is_sil_phoneme(phs[i]) or not is_sil_phoneme(phs_[-1]):
                    phs_.append(phs[i])
                elif phs_[-1] == '|' and is_sil_phoneme(phs[i]) and phs[i] != '|':
                    phs_[-1] = phs[i]
            cur_word = []
            phs_for_align = []
            phs_for_dict = set()
            for p in phs_:
                if is_sil_phoneme(p):
                    if len(cur_word) > 0:
                        phs_for_align.append('_'.join(cur_word))
                        phs_for_dict.add(' '.join(cur_word))
                        cur_word = []
                    if p not in self.txt_processor.sp_phonemes():
                        phs_for_align.append('SIL')
                else:
                    cur_word.append(p)
            
            phs_for_align = " ".join(phs_for_align)
            phs = ['|' if item == '#' else item for item in phs_]
            words="|".join([j for j in txt])
            words=["<BOS>"]+[j for j in words]+["<EOS>"]
            count=0
            ph2word=[]
            for tmp in phs:
                if tmp=='|' or tmp=='<BOS>' or tmp=='<EOS>':
                    count+=1
                    ph2word.append(count)
                    count+=1
                else:
                    ph2word.append(count)
            ph_gb_word= ["<BOS>"] + [tmp  for tmp in phs_for_align.split() if tmp!='SIL'] + ["<EOS>"]
            return " ".join(phs), txt, " ".join(words), ph2word, " ".join(ph_gb_word)
        
        
    def phone_encoder(self, ph_set):
        ph_set_fn = f"{self.processed_dir}/phone_set.json"
        if self.reset_phone_dict or not os.path.exists(ph_set_fn):
            ph_set = sorted(set(ph_set))
            json.dump(ph_set, open(ph_set_fn, 'w'), ensure_ascii=False)
            print("| Build phone set: ", ph_set)
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))
            print("| Load phone set: ", ph_set)
        return build_token_encoder(ph_set_fn)

    def word_encoder(self, word_set):
        word_set_fn = f"{self.processed_dir}/word_set.json"
        if self.reset_word_dict:
            word_set = Counter(word_set) # Counter用于对可迭代对象中元素的计数
            total_words = sum(word_set.values())
            word_set = word_set.most_common(self.word_dict_size) # most_common方法返回按计数降序排列的元素和计数的列表
            num_unk_words = total_words - sum([x[1] for x in word_set])
            word_set = ['<BOS>', '<EOS>'] + [x[0] for x in word_set]
            word_set = sorted(set(word_set))
            json.dump(word_set, open(word_set_fn, 'w'), ensure_ascii=False)
            print(f"| Build word set. Size: {len(word_set)}, #total words: {total_words},"
                  f" #unk_words: {num_unk_words}, word_set[:10]:, {word_set[:10]}.")
        else:
            word_set = json.load(open(word_set_fn, 'r'))
            print("| Load word set. Size: ", len(word_set), word_set[:10])
        return build_token_encoder(word_set_fn)
    
    def build_spk_map(self, spk_names):
        spk_map = {x: i for i, x in enumerate(sorted(list(spk_names)))}
        # assert len(spk_map) == 0 or len(spk_map) <= self.num_spk, len(spk_map)
        print(f"| Number of spks: {len(spk_map)}, spk_map: {spk_map}")
        json.dump(spk_map, open(f"{self.processed_dir}/spk_map.json", 'w'), ensure_ascii=False)
        return spk_map
    
    def preprocess_first_pass(self, item_name, txt_raw, wav_fn, others=None): # 没啥用啊，就是挪个地方。。后续优化（优化点一）
        ph, txt, word, ph2word, ph_gb_word = self.txt_to_ph(txt_raw) # 主要作用就这一块；这儿对于phonme可以直接使用aishell3中的音素而非自动生成（优化点二）
        
        # ext = os.path.splitext(wav_fn)[1]   # 获取文件后缀, .wav 可以用来加后缀，这儿不需要了。
        # new_wav_fn = f"{self.wav_processed_dir}/{item_name}{ext}"  # 
        
        os.makedirs(f"{self.processed_dir}/wav_processed", exist_ok=True)
        processed_wav_fn = f"{self.processed_dir}/wav_processed/{item_name}"
        
        # print("wav_fn: ", wav_fn) # train/wav/SSB0005/SSB00050001.wav
        # print("processed_wav_fn: ", new_wav_fn) # processed_wav_fn:  data_processed/aishell3/wav_processed/SSB00050001.wav
        
        # move_link_func = move_file
        move_link_func = link_file # 就是link_file这儿；
        
        if os.path.exists(processed_wav_fn):
            remove_file(processed_wav_fn)
        move_link_func(f"{self.raw_data_dir}/{wav_fn}", processed_wav_fn)
        return {
            'txt': txt, 'txt_raw': txt_raw, 'ph': ph,
            'word': word, 'ph2word': ph2word, 'ph_gb_word': ph_gb_word,
            'wav_fn': wav_fn,
            'processed_wav_fn': processed_wav_fn }
        
    def preprocess_second_pass(self, word, ph, spk_name, word_encoder, ph_encoder, spk_map):
        word_token = word_encoder.encode(word)
        ph_token = ph_encoder.encode(ph)
        spk_id = spk_map[spk_name]
        return {'word_token': word_token, 'ph_token': ph_token, 'spk_id': spk_id}
    
    def build_mfa_inputs(self, item, mfa_input_dir, mfa_group, temp_dir):
        item_name = item['item_name']
        wav_align_fn = item['wav_fn']
        ph_gb_word = item['ph_gb_word']
        
        mfa_input_group_dir = f'{mfa_input_dir}/{mfa_group}'
        os.makedirs(mfa_input_group_dir, exist_ok=True)
        
        # ext = os.path.splitext(wav_align_fn)[1]
        # new_wav_align_fn = f"{mfa_input_group_dir}/{item_name}{ext}"
        new_wav_align_fn = f"{mfa_input_group_dir}/{item_name}" # 这儿还是不需要重复后缀
        
        move_link_func = link_file
        if os.path.exists(new_wav_align_fn):
            remove_file(new_wav_align_fn)
 
        # print("source: ", f"{self.raw_data_dir}/{wav_align_fn}")
        # print("target: ", new_wav_align_fn)
        move_link_func(f"{self.raw_data_dir}/{wav_align_fn}", new_wav_align_fn) # 移动wav文件
        
        ph_gb_word_nosil = " ".join(["_".join([p for p in w.split("_") if not is_sil_phoneme(p)]) # 把拼音拆成声母和韵母；
                                     for w in ph_gb_word.split(" ") if not is_sil_phoneme(w)]) # 分成一个字一个字的拼音；总的来说就是去掉拼音里的静音元素;
        
        item_name_without_ext = item_name[:-4] # 去掉.wav后缀
        with open(f'{mfa_input_group_dir}/{item_name_without_ext}.lab', 'w') as f_txt:
            f_txt.write(ph_gb_word_nosil)
        return ph_gb_word_nosil, new_wav_align_fn # 音素文件，音频路径
        
    def process(self):
        
        # step1: load data:
        meta_data = list(tqdm(self.meta_data(), desc='Load meta data'))
        
        item_names = [d['item_name'] for d in meta_data]
        assert len(item_names) == len(set(item_names)), 'Key `item_name` should be Unique.'
        
        # step2: preprocess data
        phone_list = []
        word_list = []
        spk_names = set()
        # process_item = partial(self.preprocess_first_pass) # 没有参数，没有关键字，因此舍弃partial。
        args = [{
            'item_name': item_raw['item_name'],
            'txt_raw': item_raw['txt'],
            'wav_fn': item_raw['wav_fn'],
            'others': item_raw.get('others', None)
        } for item_raw in meta_data]
        
        items = []
        for item_, (idx, item) in zip(meta_data, multiprocess_run_tqdm(self.preprocess_first_pass, args, num_workers=1, desc='Preprocess')): #multiprocess_run_tqdm额外返回一个编号
            if item is not None:
                # 这儿就是纯在复制：
                item_.update(item) # update函数通常用于更新字典（dict）的内容，用后面那个字典的键值对去更新到前面那个字典；更新：相同的键修改，不同的键二者都保留；
                item = item_
                item['id'] = idx
                item['spk_name'] = item.get('spk_name', '<SINGLE_SPK>')
                item['others'] = item.get('others', None)
                items.append(item)
                
                # 统计：
                phone_list += item['ph'].split(" ") 
                word_list += item['word'].split(" ") # 这儿是起到了一个统计数量的作用吗？
                spk_names.add(item['spk_name'])
                
        # step3: encoder
        ph_encoder, word_encoder = self.phone_encoder(phone_list), self.word_encoder(word_list)
        spk_map = self.build_spk_map(spk_names)
        args = [{
            'ph': item['ph'], 'word': item['word'], 'spk_name': item['spk_name'],
            'word_encoder': word_encoder, 'ph_encoder': ph_encoder, 'spk_map': spk_map
        } for item in items]
        
        for idx, item_new_kv in multiprocess_run_tqdm(self.preprocess_second_pass, args, desc='Add encoded tokens'):
            items[idx].update(item_new_kv)
            
        # 此时此刻，items就是一个具有以下键值对的字典的列表，是符合需求的元数据：
        
        # print("items keys: ", items[0].keys()) # dict_keys(['item_name', 'wav_fn', 'txt', 'phones', 'spk_name', ...
        # print("item_name: ", items[0]['item_name']) # SSB00050001.wav
        # print("wav_fn: ", items[0]['wav_fn']) # train/wav/SSB0005/SSB00050001.wav
        # print("txt: ", items[0]['txt']) # 广州女大学生登山失联四天警方找到疑似女尸
        # print("phones: ", items[0]['phones']) # guang3 zhou1 nv3 da4 xue2 sheng1 deng1 shan1 shi1 lian2 si4 tian1 jing3 fang1 zhao3 dao4 yi2 si4 nv3 shi1
        # print("spk_name: ", items[0]['spk_name']) # SSB0005
        # print("txt_raw: ", items[0]['txt_raw']) # 广州女大学生登山失联四天警方找到疑似女尸
        # print("ph: ", items[0]['ph']) # <BOS> g uang3 | zh ou1 | n v3 | d a4 | x ue2 | sh eng1 | d eng1 | sh an1 | sh i1 | ....
        # print("word: ", items[0]['word']) # <BOS> 广 | 州 | 女 | 大 | 学 | 生 | 登 | 山 | 失 | 联 | 四 | 天 | 警 | 方 | 找 | 到 | 疑 | 似 | 女 | 尸 <EOS>
        # print("ph2word: ", items[0]['ph2word']) # [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16, ....
        # print("ph_gb_word: ", items[0]['ph_gb_word']) # <BOS> g_uang3 zh_ou1 n_v3 d_a4 x_ue2 sh_eng1 d_eng1 sh_an1 sh_i1 l_ian2 s_i4 t_ian1
        # print("processed_wav_fn: ", items[0]['processed_wav_fn']) # data_processed/aishell3/wav_processed/SSB00050001.wav
        # print("id: ", items[0]['id']) # 0
        # print("others: ", items[0]['others']) # None
        # print("word_token: ", items[0]['word_token']) # [3, 40, 4, 38, 4, 28, 4, 25, 4, 30, 4, 58, 4, 61, 4, 37, 4, 27, 4, 68, 4, 23, 4, 26, 4, ....
        # print("ph_token: ", items[0]['ph_token']) # [3, 29, 68, 83, 82, 52, 83, 47, 76, 83, 18, 5, 83, 79, 69, 83, 59, 26, 83, 18, 26, 83, 59, ...
        # print("spk_id: ", items[0]['spk_id']) # 0
        
        # step3: prepare mfa input data：
        remove_file(self.mfa_input_dir)
        mfa_dict = set()
        # group MFA inputs for better parallelism
        mfa_groups = [i // self.nsample_per_mfa_group for i in range(len(items))]
        
        # print("mfa_groups: ", mfa_groups) # mfa_groups:  [0, 0, 1, 1, 2, 2, 3]
        if self.mfa_group_shuffle:
            random.seed(self.seed)
            random.shuffle(mfa_groups)
        args = [{
            'item': item, 'mfa_input_dir': self.mfa_input_dir,
            'mfa_group': mfa_group, 'temp_dir': self.temp_dir_for_process
        } for item, mfa_group in zip(items, mfa_groups)]
        
        for i, (ph_gb_word_nosil, new_wav_align_fn) in multiprocess_run_tqdm(self.build_mfa_inputs, args, desc='Build MFA data'):
            items[i]['wav_align_fn'] = new_wav_align_fn
            for w in ph_gb_word_nosil.split(" "):
                mfa_dict.add(f"{w} {w.replace('_', ' ')}") # mfa_dict里装的是所有需要做对齐的拼音元素，中间用空格隔开；
                
        mfa_dict = sorted(mfa_dict)
        with open(f'{self.processed_dir}/mfa_dict.txt', 'w') as f:
            f.writelines([f'{l}\n' for l in mfa_dict]) 
            
        with open(f"{self.processed_dir}/metadata.json", 'w') as f:
            f.write(re.sub(r'\n\s+([\d+\]])', r'\1', json.dumps(items, ensure_ascii=False, sort_keys=False, indent=1)))
        remove_file(self.temp_dir_for_process)
       
    # 推理的时候用 
    def load_dict(self):
        ph_encoder = build_token_encoder(f'{self.processed_dir}/phone_set.json')
        word_encoder = build_token_encoder(f'{self.processed_dir}/word_set.json')
        return ph_encoder, word_encoder
    
    def load_spk_map(self):
        spk_map_fn = f"{self.processed_dir}/spk_map.json"
        spk_map = json.load(open(spk_map_fn, 'r'))
        return spk_map
            
if __name__ == '__main__':
    BasePreprocessor().process()