# coding=gbk

import os
import numpy as np
import torch
import sys
import shutil
import pandas as pd
from tqdm import tqdm
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# from data_gen.tts.base_preprocess import BasePreprocessor
from data_gen.tts.base_preprocess_ai3 import BasePreprocessor
from inference.tts.base_tts_infer import BaseTTSInfer
from inference.tts.infer_utils import get_align_from_mfa_output, extract_f0_uv
from modules.speech_editing.spec_denoiser.spec_denoiser import GaussianDiffusion
from modules.speech_editing.spec_denoiser.diffnet import DiffNet
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.spec_aug.time_mask import generate_time_mask
from utils.text.text_encoder import is_sil_phoneme
from resemblyzer import VoiceEncoder
from utils.audio import librosa_wav2spec
from inference.tts.infer_utils import get_words_region_from_origintxt_region, parse_region_list_from_str

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
}


class SpecDenoiserInfer(BaseTTSInfer):
    def __init__(self, hparams, device=None):
        if device is None:
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            device = 'cpu'
        self.hparams = hparams
        self.device = device
        self.data_dir = hparams['binary_data_dir']
        self.preprocessor = BasePreprocessor()
        self.ph_encoder, self.word_encoder = self.preprocessor.load_dict()
        self.spk_map = self.preprocessor.load_spk_map()
        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder = self.build_vocoder()
        self.vocoder.eval()
        self.vocoder.to(self.device)
        self.spk_embeding = VoiceEncoder(device='cpu')

    def build_model(self):
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        load_ckpt(model, hparams['work_dir'], 'model')
        model.to(self.device)
        model.eval()
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)

        # Forward the edited txt to the encoder
        edited_txt_tokens = sample['edited_txt_tokens']
        mel = sample['mel']
        mel2ph = sample['mel2ph']
        mel2word = sample['mel2word']
        dur = sample['dur']
        ph2word = sample['ph2word']
        edited_ph2word = sample['edited_ph2word']
        f0 = sample['f0']
        uv = sample['uv']
        words_region = sample['words_region']
        edited_words_region = sample['edited_words_region']
        text = sample['text']

        edited_word_idx = words_region[0]
        changed_idx = edited_words_region[0]

        ret = {}
        encoder_out = self.model.fs.encoder(edited_txt_tokens)  # [B, T, C]
        src_nonpadding = (edited_txt_tokens > 0).float()[:, :, None]
        style_embed = self.model.fs.forward_style_embed(sample['spk_embed'], None)
        
        masked_dur = torch.zeros_like(edited_ph2word).to(self.device)
        masked_dur[:, :ph2word[ph2word<edited_word_idx[0]].size(0)] = dur[:, :ph2word[ph2word<edited_word_idx[0]].size(0)]
        if ph2word.max() > edited_word_idx[1]:
            masked_dur[:, -ph2word[ph2word>edited_word_idx[1]].size(0):] = dur[:, -ph2word[ph2word>edited_word_idx[1]].size(0):]
        # Forward duration model to get the duration and mel2ph for edited text seq (Note that is_editing is set as False to get edited_mel2ph)
        dur_inp = (encoder_out + style_embed) * src_nonpadding
        masked_mel2ph = mel2ph
        masked_mel2ph[(mel2word>=edited_word_idx[0]) & (mel2word<=edited_word_idx[1])] = 0
        time_mel_masks_orig = torch.zeros_like(mel2ph).to(self.device)
        time_mel_masks_orig[(mel2word>=edited_word_idx[0]) & (mel2word<=edited_word_idx[1])] = 1.0
        edited_mel2ph = self.model.fs.forward_dur(dur_inp, time_mel_masks_orig, masked_mel2ph, edited_txt_tokens, ret, masked_dur=masked_dur, use_pred_mel2ph=True)
        edited_mel2word = torch.Tensor([edited_ph2word[0].numpy()[p - 1] for p in edited_mel2ph[0]]).to(self.device)[None, :]
        length_edited = edited_mel2word[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].size(0) - mel2word[(mel2word>=edited_word_idx[0]) & (mel2word<=edited_word_idx[1])].size(0)
        edited_mel2ph_ = torch.zeros((1, mel2ph.size(1)+length_edited)).to(self.device)
        head_idx = mel2word[mel2word<edited_word_idx[0]].size(0)
        tail_idx = mel2word[mel2word<=edited_word_idx[1]].size(0) + length_edited
        edited_mel2ph_[:, :head_idx] = mel2ph[:, :head_idx]
        edited_mel2ph_[:, head_idx:tail_idx] = edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])]
        if mel2word.max() > edited_word_idx[1]:
            edited_mel2ph_[:, tail_idx:] = mel2ph[mel2word>edited_word_idx[1]] - mel2ph[mel2word>edited_word_idx[1]].min() + edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max() + 2
        edited_mel2ph = edited_mel2ph_.long()

        # Get masked mel by concating the head and tial of the original mel
        length_edited = edited_mel2word[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].size(0) - mel2word[(mel2word>=edited_word_idx[0]) & (mel2word<=edited_word_idx[1])].size(0)
        head_idx = mel2word[mel2word<edited_word_idx[0]].size(0)
        tail_idx = mel2word[mel2word<=edited_word_idx[1]].size(0) + length_edited

        # Create masked ref mel
        ref_mels = torch.zeros((1, edited_mel2ph.size(1), mel.size(2))).to(self.device)
        T = min(ref_mels.size(1), mel.size(1))
        ref_mels[:, :head_idx, :] = mel[:, :head_idx, :]
        ref_mels[:, tail_idx:, :] = mel[mel2word>edited_word_idx[1]]

        # Get masked frame-level f0 and uv (pitch info)
        edited_f0 = torch.zeros((1, edited_mel2ph.size(1))).to(self.device)
        edited_uv = torch.zeros((1, edited_mel2ph.size(1))).to(self.device)
        edited_f0[:, :head_idx] = f0[:, :head_idx]
        edited_f0[:, tail_idx:] = f0[mel2word>edited_word_idx[1]]
        edited_uv[:, :head_idx] = uv[:, :head_idx]
        edited_uv[:, tail_idx:] = uv[mel2word>edited_word_idx[1]]

        # Create time mask
        time_mel_masks = torch.zeros((1, edited_mel2ph.size(1), 1)).to(self.device)
        time_mel_masks[:, head_idx:tail_idx] = 1.0
        
        with torch.no_grad():
            output = self.model(edited_txt_tokens, time_mel_masks=time_mel_masks, mel2ph=edited_mel2ph, spk_embed=sample['spk_embed'],
                       ref_mels=ref_mels, f0=edited_f0, uv=edited_uv, energy=None, infer=True, use_pred_pitch=True)
            mel_out = output['mel_out'] * time_mel_masks + ref_mels * (1-time_mel_masks)
            wav_out = self.run_vocoder(mel_out)
            wav_gt = self.run_vocoder(sample['mel'])
            # item_name = sample['item_name'][0]
            # np.save(f'inference_acl/mel2ph/{item_name}',output['mel2ph'].cpu().numpy()[0])

        wav_out = wav_out.cpu().numpy()
        wav_gt = wav_gt.cpu().numpy()
        mel_out = mel_out.cpu().numpy()
        mel_gt = sample['mel'].cpu().numpy()
        masked_mel_out = ref_mels.cpu().numpy()
        masked_mel_gt = (sample['mel'] * time_mel_masks_orig[:, :, None]).cpu().numpy()

        return wav_out[0], wav_gt[0], mel_out[0], mel_gt[0], masked_mel_out[0], masked_mel_gt[0]

    def preprocess_input(self, inp):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        # Get ph for original txt
        preprocessor = self.preprocessor
        text_raw = inp['text']
        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', '<SINGLE_SPK>')
        ph, txt, words, ph2word, ph_gb_word = preprocessor.txt_to_ph(
            text_raw)
            # preprocessor.txt_processor, text_raw)
        ph_token = self.ph_encoder.encode(ph)
        # Get ph for edited txt
        edited_text_raw = inp['edited_text']
        edited_ph, _, edited_words, edited_ph2word, _ = preprocessor.txt_to_ph(
            edited_text_raw)
            # preprocessor.txt_processor, edited_text_raw)
        edited_ph_token = self.ph_encoder.encode(edited_ph)

        # Get words_region
        words = words.split(' ')
        edited_words = edited_words.split(' ')
        # print("6666666666666666666666666")
        # print(inp['region'])
        region, edited_region = parse_region_list_from_str(inp['region']), parse_region_list_from_str(
            inp['edited_region'])
        # print("9999999999999999999999999")
        # print(region)
        words_region = get_words_region_from_origintxt_region(words, region)
        edited_words_region = get_words_region_from_origintxt_region(edited_words, edited_region)

        # Generate forced alignment
        wav = inp['wav']
        mel = inp['mel']
        mfa_textgrid = inp['mfa_textgrid']
        mel2ph, dur = get_align_from_mfa_output(mfa_textgrid, ph, ph_token, mel)
        mel2word = [ph2word[p - 1] for p in mel2ph]  # [T_mel]

        # Extract frame-level f0 and uv (pitch info)
        f0, uv = extract_f0_uv(wav, mel)

        item = {'item_name': item_name, 'text': txt, 'ph': ph,
                'ph2word': ph2word, 'edited_ph2word': edited_ph2word,
                'ph_token': ph_token, 'edited_ph_token': edited_ph_token,
                'words_region': words_region, 'edited_words_region': edited_words_region,
                'mel2ph': mel2ph, 'mel2word': mel2word, 'dur': dur,
                'f0': f0, 'uv': uv,
                'mel': mel, 'wav': wav}
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        ph2word = torch.LongTensor(item['ph2word'])[None, :].to(self.device)
        edited_ph2word = torch.LongTensor(item['edited_ph2word'])[None, :].to(self.device)
        mel2ph = torch.LongTensor(item['mel2ph'])[None, :].to(self.device)
        dur = torch.LongTensor(item['dur'])[None, :].to(self.device)
        mel2word = torch.LongTensor(item['mel2word'])[None, :].to(self.device)
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        edited_txt_tokens = torch.LongTensor(item['edited_ph_token'])[None, :].to(self.device)
        # spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)

        # masked prediction related
        mel = torch.FloatTensor(item['mel'])[None, :].to(self.device)
        wav = torch.FloatTensor(item['wav'])[None, :].to(self.device)

        # get spk embed
        spk_embed = self.spk_embeding.embed_utterance(item['wav'].astype(float))
        spk_embed = torch.FloatTensor(spk_embed[None, :]).to(self.device)

        # get frame-level f0 and uv (pitch info)
        f0 = torch.FloatTensor(item['f0'])[None, :].to(self.device)
        uv = torch.FloatTensor(item['uv'])[None, :].to(self.device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'ph2word': ph2word,
            'edited_ph2word': edited_ph2word,
            'mel2ph': mel2ph,
            'mel2word': mel2word,
            'dur': dur,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'edited_txt_tokens': edited_txt_tokens,
            'words_region': item['words_region'],
            'edited_words_region': item['edited_words_region'],
            # 'spk_ids': spk_ids,
            'mel': mel,
            'wav': wav,
            'spk_embed': spk_embed,
            'f0': f0,
            'uv': uv
        }
        return batch

    @classmethod
    def example_run(cls, dataset_info):
        from utils.commons.hparams import set_hparams
        from utils.commons.hparams import hparams as hp
        from utils.audio.io import save_wav
        from utils.plot.plot import plot_mel

        set_hparams()
        infer_ins = cls(hp)

        def infer_one(data_info):
            wav2spec_res = librosa_wav2spec(data_info['wav_fn_orig'], fmin=55, fmax=7600, sample_rate=22050)
            inp = {
                'item_name': data_info['item_name'],
                'text': data_info['text'],
                'edited_text': data_info['edited_text'],
                'region': data_info['region'],
                'edited_region': data_info['edited_region'],
                'mfa_textgrid': data_info['mfa_textgrid'],
                'mel': wav2spec_res['mel'],
                'wav': wav2spec_res['wav']
            }
            
            print("===================================")
            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            
            from scipy import ndimage
            image = inp['mel']
            angle = 90 # 正逆时针，负顺时针
            rotated_image = ndimage.rotate(image, angle, reshape=False)
            
            plt.figure()
            plt.imshow(rotated_image, aspect=None) # 'equal' 强制图像的纵横比例相等，使图像在显示时不会出现拉伸或压缩。None：使用数据坐标的纵横比例，这意味着图像的纵横比例可能会在显示时发生变化。注意二者一个有引号，一个没有。
            plt.savefig('mel_fig/mel.jpg')
            print("==========MEL_FIGS_SAVED===========")
            
            
            wav_out, wav_gt, mel_out, mel_gt, masked_mel_out, masked_mel_gt = infer_ins.infer_once(inp)
            os.makedirs(f'inference/out', exist_ok=True)
            save_wav(wav_out, f'inference/out/{inp["item_name"]}.wav', hp['audio_sample_rate'])
            save_wav(wav_gt, f'inference/out/{inp["item_name"]}_ref.wav', hp['audio_sample_rate'])
            return 1

        os.makedirs('infer_out', exist_ok=True)
        for item in dataset_info:
            infer_one(item)


def load_dataset_info(file_path):
    dataset_frame = pd.read_csv(file_path, encoding='gbk') # 读取中文需要指定编码格式 jyh
    dataset_info = []
    for index, row in dataset_frame.iterrows():
        row_info = {}
        row_info['item_name'] = row['item_name']
        row_info['text'] = row['text']
        row_info['edited_text'] = row['edited_text']
        row_info['wav_fn_orig'] = row['wav_fn_orig']
        row_info['edited_region'] = row['edited_region']
        row_info['region'] = row['region']
        dataset_info.append(row_info)
    return dataset_info


# preprocess data with forced alignment
def data_preprocess(file_path, input_directory, dictionary_path, acoustic_model_path, output_directory, align=True):
    print(input_directory)
    assert os.path.exists(input_directory) 
    assert os.path.exists(acoustic_model_path)
    assert f"{input_directory},{dictionary_path},{acoustic_model_path}"
        
    data_info = file_path # dataset_info是一个列表，每一个元素对应csv中的一行;
    data_info['mfa_textgrid'] = f'{output_directory}/{data_info["item_name"]}.TextGrid'
    
    # print(data_info)
    # print("222")
    # from data_gen.tts.txt_processors.en import TxtProcessor
    # txt_processor = TxtProcessor()

    basepreprocessor = BasePreprocessor()
    # gen  .lab file
    def gen_forced_alignment_info(data_info):
        # *_, ph_gb_word = BasePreprocessor.txt_to_ph(txt_processor, data_info['text'])
        # print(data_info['text'])
        # print("444")
        *_, ph_gb_word = basepreprocessor.txt_to_ph(data_info['text']) # 这儿被我改成成员函数了
        # print(ph_gb_word)
        # print("555")
        tg_fn = f'{input_directory}/{data_info["item_name"]}.lab'
        ph_gb_word_nosil = " ".join(["_".join([p for p in w.split("_") if not is_sil_phoneme(p)])
                                     for w in ph_gb_word.split(" ") if not is_sil_phoneme(w)])
        with open(tg_fn, 'w') as f_txt:
            f_txt.write(ph_gb_word_nosil)
        with open(dictionary_path, 'r') as f:  # update mfa dict for unseen word
            lines = f.readlines()
        with open(dictionary_path, 'a+') as f:
            for item in ph_gb_word_nosil.split(" "):
                item = item + '\t' + ' '.join(item.split('_')) + '\n'
                if item not in lines:
                    f.writelines([item])
    # print("333")
    # for item in dataset_info:
        # print(item['text'])
    gen_forced_alignment_info(data_info)
    item_name, wav_fn_orig = data_info['item_name'], data_info['wav_fn_orig']
    os.system(f'cp -f {wav_fn_orig} inference/audio/{item_name}.wav')
    # print("999")
    # return

    print("Generating forced alignments with mfa. Please wait for about several minutes.")
    mfa_out = output_directory
    if os.path.exists(mfa_out):
        shutil.rmtree(mfa_out)
    command = ' '.join(
        ['mfa align -j 4 --clean', input_directory, dictionary_path, acoustic_model_path, output_directory])
    # mfa align -j 4 --clean inference/audio data/processed/aishell3/mfa_dict.txt data/processed/aishell3/mfa_model.zip inference/audio/mfa_out
    # print(command)
    os.system(command)

    return data_info

if __name__ == '__main__':
    # you can use 'align' to choose whether using MFA during preprocessing
    # test_file_path = 'inference/example.csv'
    import subprocess
    commands = [f"export PYTHONPATH=."]
    for i in range(len(commands)):
        result = subprocess.run(commands[i], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # 输出命令的标准输出和标准错误
    if len(result.stdout) != 0:
        print("Output:", result.stdout)
    if len(result.stderr) != 0:
        print("Error:", result.stderr)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--text')
    parser.add_argument('--edited_text')
    parser.add_argument('--region')
    parser.add_argument('--edited_region')
    parser.add_argument('--exp_name spec_denoiser_ai3', type=str, default="spec_denoiser_ai3")
    parser.add_argument('--config', type=str, default="egs/spec_denoiser_aishell3.yaml")
    args = parser.parse_args()
    text_and_index = {
        'id': 0,
        'item_name': 'original_audio',
        'wav_fn_orig': 'inference/audio_backup/original_audio.wav',
        'text': args.text,
        'edited_text': args.edited_text,
        'region': args.region,
        'edited_region': args.edited_region
    }
    test_wav_directory = 'inference/audio'
    dictionary_path = 'data/processed/aishell3/mfa_dict.txt'
    acoustic_model_path = 'data/processed/aishell3/mfa_model.zip'
    output_directory = 'inference/audio/mfa_out'
    # os.system('rm -r inference/audio')
    # os.makedirs(f'inference/audio', exist_ok=True)
    
    dataset_info = data_preprocess(text_and_index, test_wav_directory, dictionary_path, acoustic_model_path,
                                   output_directory, align=True)
    # print(dataset_info)
    SpecDenoiserInfer.example_run([dataset_info])


# {'item_name': 'SSB0033_zh', 'text': '刘嘉玲虽然来不及准备贺礼', 'edited_text': '梁朝伟虽然来不及准备贺礼', 'wav_fn_orig': 'inference/audio_backup/original_audio.wav', 'edited_region': '[1,3]', 'region': '[1,3]', 'mfa_textgrid': 'inference/audio/mfa_out/SSB0033_zh.TextGrid'}
# {'id': 0, 'item_name': 'original_audio', 'wav_fn_orig': 'inference/audio_backup/original_audio.wav', 'text': '我是南开大学电子信息与光学工程学院的聂鑫', 'edited_text': '我是南开大学计算机学院的聂鑫', 'region': '[7, 15]', 'edited_region': '[7, 9]', 'mfa_textgrid': 'inference/audio/mfa_out/original_audio.TextGrid'}

# {'id': 0, 'item_name': 'original_audio', 'wav_fn_orig': 'inference/audio_backup/original_audio.wav', 'text': '我是南开大学电子信息与光学工程学院的聂鑫', 'edited_text': '我是南开大学计算机学院的聂鑫', 'region': '[7, 15]', 'edited_region': '[7, 9]', 'mfa_textgrid': 'inference/audio/mfa_out/original_audio.TextGrid'}
# [{'item_name': 'SSB0033_zh', 'text': '刘嘉玲虽然来不及准备贺礼', 'edited_text': '梁朝伟虽然来不及准备贺礼', 'wav_fn_orig': 'inference/audio_backup/original_audio.wav', 'edited_region': '[1,3]', 'region': '[1,3]', 'mfa_textgrid': 'inference/audio/mfa_out/SSB0033_zh.TextGrid'}]