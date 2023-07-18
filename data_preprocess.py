from src.utils.get_file import Get_img_paths
from multiprocessing.pool import Pool
from functools import partial
from tqdm import tqdm
import subprocess,cv2,os
from src.utils.init_path import init_path
from src.utils.preprocess import CropAndExtract
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ['CUDA_VISIBLE_DEVICES'] ="6"




template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}' #for save audio


def extract_audios_from_videos_multi(video_paths, audio_save_dir, video_org_dir):
    os.makedirs(audio_save_dir, exist_ok= True)
    with Pool(processes=1) as p: 
        with tqdm(total=len(video_paths)) as pbar:
            func = partial(extract_audios_from_videos, audio_save_dir = audio_save_dir , video_org_dir = video_org_dir)
            for v in p.imap_unordered(func, video_paths):
                pbar.update()


def extract_audios_from_videos(vfile, audio_save_dir, video_org_dir):
    try:
        vidname = os.path.basename(vfile).split('.')[0]

        fulldir = vfile.replace(video_org_dir, audio_save_dir) 
        fulldir = fulldir.split('.')[0]
        os.makedirs(fulldir, exist_ok=True)

        wavpath = os.path.join(fulldir, 'audio.wav')
        command = template.format(vfile, wavpath)
        subprocess.call(command, shell=True)                                                
    except Exception as e:
        print(e) 
        return



def mp_handler(job):
    vfile, save_root, org_root, gpu_id = job
    try:
        print('processing ====>{}, current_gpu:{}, process indx {}'.format(vfile, gpu_id, os.getpid()))
        fa[gpu_id].generate_for_train(vfile, save_root, org_root, 'crop',\
                                        source_image_flag=False, pic_size=size)
    except Exception as e:
        print(e) 
        return



if __name__ == '__main__':
    
    #预处理结果保存的路径
    preprocess_save_dir = '/metahuman/wyt/debug/'
    os.makedirs(preprocess_save_dir, exist_ok= True)
    
    
    #step1:提取audio
    input_dir = '/metahuman/data/2dhighresolution_230606/数字人高清视频数据_230606/数字人高清视频数据_230608_25fps'
    video_org_dir = '/metahuman/data/'
    audio_save_dir =  preprocess_save_dir +  '/audio/'
    video_paths = Get_img_paths(input_dir, ext = 'mp4')
    extract_audios_from_videos_multi(video_paths, audio_save_dir, video_org_dir)

    #step2： 这将花费相当长的时间
    #读取预处理模型
    checkpoint_dir = './checkpoints'
    config_dir = './src/config/'
    size = 256 
    old_version = False
    preprocess = 'crop'
    sadtalker_paths = init_path(checkpoint_dir, config_dir, size, old_version, preprocess)
    ngpu = 1 #采用GPU的数量
    fa = [CropAndExtract(sadtalker_paths, device='cuda:{}'.format(id)) for id in range(ngpu)]  #构建GPU

    pose_save_dir =  preprocess_save_dir +  '/pose/'  #保存ρ的路径
    os.makedirs(preprocess_save_dir, exist_ok= True)
    jobs = [(vfile, pose_save_dir, video_org_dir, i%ngpu) for i, vfile in enumerate(video_paths)]
    p = ThreadPoolExecutor(ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]


