import os
from multiprocessing.pool import Pool
from functools import partial
import src.utils.audio as audio
import numpy as np 
from tqdm import tqdm
from src.utils.get_file import Get_img_paths


def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav

def parse_audio_length(audio_length, sr, fps):
    #time = audio_length / sr  #视频的长度
    #那么对应的图像共有： num_frames = time * fps = audio_length / sr * fps
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames


def save_orig_mel(wavpath):
    try:
        wav = audio.load_wav(wavpath, 16000) 
        wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)    #将音频重采样并对准到25fps
        wav = crop_pad_audio(wav, wav_length)
        orig_mel = audio.melspectrogram(wav).T    #得到特征
        diranme = os.path.dirname(wavpath)
        save_path = diranme.replace(org_dir,save_dir)
        os.makedirs(save_path, exist_ok= True)
        save_path = save_path + '/orig_mel.npy'
        np.save(save_path,orig_mel)
    except:
        return 


inputs_path = '/metahuman/wyt/preprocess/audio/2dhighresolution_230606_wav.txt'  #wav_dir
wav_paths = Get_img_paths(inputs_path)

org_dir = '/metahuman/wyt/preprocess/audio/'
save_dir = '/metahuman/wyt/preprocess/orig_mel/'

with Pool(processes=16) as p:  #os.cpu_count()
    with tqdm(total=len(wav_paths)) as pbar:
        func = partial(save_orig_mel)
        for v in p.imap_unordered(func, wav_paths):
            pbar.update()
