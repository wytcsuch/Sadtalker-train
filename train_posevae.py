from glob import glob
import shutil
import torch
from torch.utils.data import Dataset
from time import  strftime
import os, sys, time
from argparse import ArgumentParser
import random
import src.utils.audio as audio
import cv2
import numpy as np
from scipy.io import loadmat, savemat

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from src.audio2pose_models.audio2pose import Audio2Pose  #姿态模型
from yacs.config import CfgNode as CN
from src.loss import GANLoss 
from src.utils.safetensor_helper import load_x_from_safetensor
import safetensors  
import safetensors.torch
import time
import logging
from src.utils.logger import create_logger
logger = logging.getLogger("poseVae_train") 
from src.utils.get_file import Get_img_paths


class My_Dataset(Dataset):
    def __init__(self,input_path , syncnet_T = 32, syncnet_mel_step_size = 16):
        self.all_videos = Get_img_paths(input_path)
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size

    #得到帧数的id
    def get_frame_id(self, frame):
        return int(os.path.basename(frame).split('_')[-1].split('.')[0])  #

    #得到window
    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidpath = os.path.dirname(start_frame)
        vidname = vidpath.split('/')[-1]

        window_fnames = []
        end_id = start_id + self.syncnet_T
        for frame_id in range(start_id, end_id):
            frame = os.path.join(vidpath, '{}.png'.format(vidname + '_' + str(frame_id).zfill(6)))
            if not os.path.isfile(frame):  
                return None,None,None
            window_fnames.append(frame)
        return window_fnames, start_id,end_id

    #读取window的图像
    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            h,w,c = img.shape
            if h != 256 or w != 256:
                img = cv2.resize(img, (256,256))
            window.append(img)
        return window

    #获取某一帧的audio
    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(25)))
        
        end_idx = start_idx + self.syncnet_mel_step_size

        seq = list(range(start_idx, end_idx))
        seq = [ min(max(item, 0), spec.shape[0]-1) for item in seq ]
        return spec[seq, :]

    #获取window内的audio
    def get_segmented_mels(self, spec, start_frame):
        mels = []
        start_frame_num = start_frame + 1
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + self.syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != self.syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)
        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))
        return x


    def parse_audio_length(self,audio_length, sr, fps):
        #time = audio_length / sr  #视频的长度
        #那么对应的图像共有： num_frames = time * fps = audio_length / sr * fps
        bit_per_frames = sr / fps

        num_frames = int(audio_length / bit_per_frames)
        audio_length = int(num_frames * bit_per_frames)

        return audio_length, num_frames


    def crop_pad_audio(self, wav, audio_length):
        if len(wav) > audio_length:
            wav = wav[:audio_length]
        elif len(wav) < audio_length:
            wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
        return wav


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            #选择一个视频 找到对应的audio
            video_dir, label = self.all_videos[idx].split(' ')  
            dirname = video_dir.split('/')[-1]
            image_paths = glob(os.path.join(video_dir, '*.png'))  #图像的路径

            #读取音频并进行预处理
            start_time = time.time()
            try:
                wavpath = os.path.join(video_dir.replace('/images/','/orig_mel/'), 'orig_mel.npy')
                # wav = audio.load_wav(wavpath, 16000) 
                # wav_length, num_frames = self.parse_audio_length(len(wav), 16000, 25)    #将音频重采样并对准到25fps
                # wav = self.crop_pad_audio(wav, wav_length)
                # start_time = time.time()
                # orig_mel = audio.melspectrogram(wav).T    #得到特征
                orig_mel = np.load(wavpath)
            except Exception as e:
                print('not exist orig_mel:',wavpath )
                idx = random.randint(0, len(self.all_videos)-1)
                continue

            #读取pose
            pose_path = os.path.join(video_dir.replace('/images/','/pose/'), dirname + '.mat') #
            try:
                pose = loadmat(pose_path)
                coeff_3dmm = pose['coeff_3dmm']
                if  coeff_3dmm.shape[0] != len(image_paths):
                    print('mismatch coeff_3dmm and len(image_paths)', pose_path)
            except Exception as e:
                # print('pose_path not exists:', pose_path)
                idx = random.randint(0, len(self.all_videos)-1)
                continue
            
            start_time = time.time()
            ##随机选取一帧,得到窗口并读取图片
            img_path = random.choice(image_paths)  #随机选取一帧
            window_fnames, start_id, end_id = self.get_window(img_path)  #
            if window_fnames is None:
                continue
            # window = self.read_window(window_fnames)   
            # if window is None:  #读取的图像有误
            #     continue
            
            #得到winow起始帧的wav以及整个window的wav
            # mel = self.crop_audio_window(orig_mel.copy(), 1)  #起始帧的mel
            # if (mel.shape[0] != self.syncnet_mel_step_size):
            #     continue
            indiv_mels = self.get_segmented_mels(orig_mel.copy(), start_id)  #整个window的mel
            if indiv_mels is None: 
                continue
            if indiv_mels.shape != (self.syncnet_T, 80, self.syncnet_mel_step_size):
                # print('indiv_mels mismatch', video_dir)
                continue

            #得到第一帧的ceoff_3dmm以及整个window的ceoff_3dmm
            first_coeff_3dmm = np.expand_dims(coeff_3dmm[0] , 0)  #ρ0
            window_coeff_3dmm = coeff_3dmm[start_id:end_id]  #
            ref_coeff = np.repeat(first_coeff_3dmm, self.syncnet_T, axis=0)  #
            if ref_coeff.shape != (self.syncnet_T,73):
                # print('ref_coeff mismatch', video_dir)
                continue
            if window_coeff_3dmm.shape !=(self.syncnet_T,73):
                # print('window_coeff_3dmm mismatch', video_dir)
                continue


            # window = self.prepare_window(window)  #预处理图片
            first_coeff_3dmm = torch.FloatTensor(first_coeff_3dmm)
            window_coeff_3dmm = torch.FloatTensor(window_coeff_3dmm)
            indiv_mels = torch.FloatTensor(indiv_mels)
            label = int(label)
            return indiv_mels, ref_coeff, window_coeff_3dmm ,label




if __name__ == '__main__':
    parser = ArgumentParser()  
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--checkpoint_dir", default='./checkpoints/SadTalker_V0.0.2_256.safetensors', help="path to output")
    parser.add_argument("--batch_size", type=int, default=64,  help="train batch size")
    parser.add_argument('--save_dir', type=str, default='./result_pose')
    parser.add_argument('--save_name', type=str, default='exp1')
    parser.add_argument('--interval', type=int, default=25)  #print interval
    parser.add_argument('--save_interval', type=int, default=500)  #save model interval
    parser.add_argument('--num_workers', type=int, default=6)  #
    parser.add_argument('--num_class', type=int, default=46)  #
    parser.add_argument('--train_data_path', type=str, default='')  #
    args = parser.parse_args()


    device = "cuda"
    current_root_path = './'
    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    #读取姿势yaml文件
    fcfg_pose = open(sadtalker_paths['audio2pose_yaml_path'])
    cfg_pose = CN.load_cfg(fcfg_pose)
    cfg_pose['DATASET']['NUM_CLASSES'] = args.num_class
    cfg_pose.freeze()
    

    #姿态模型
    audio2pose_model = Audio2Pose(cfg_pose, None, device=device)
    audio2pose_model = audio2pose_model.to(device)
    #读取预训练的模型
    dicts = {}
    try:
        if sadtalker_paths['use_safetensor']:
            checkpoints = safetensors.torch.load_file(sadtalker_paths['checkpoint'])
            match = load_x_from_safetensor(checkpoints, 'audio2pose')
            for key in match:
                if key.startswith('audio_encoder.audio_encoder'):
                    dicts[key[14:]] = match[key]
            audio2pose_model.audio_encoder.load_state_dict(dicts)  #只读取audio_encoder
            print('load pretrained model successfully')
        else:
            load_cpk(sadtalker_paths['audio2pose_checkpoint'], model=audio2pose_model, device=device)
    except:
        raise Exception("Failed in loading audio2pose_checkpoint")

    worspace_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(worspace_path, exist_ok= True)
    logger = create_logger(output_dir=worspace_path, dist_rank=0, name="loggers")

    #构建训练datasets
    train_dataset = My_Dataset(input_path = args.train_data_path)
    bs = args.batch_size

    train_data_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=bs, 
                    shuffle=True,
                    num_workers= args.num_workers)

    D_params = list(audio2pose_model.netD_motion.parameters())
    G_params = list(audio2pose_model.netG.parameters()) 

    optimizer_G = torch.optim.Adam(G_params, lr=1e-4, betas=(0.9, 0.99))
    optimizer_D = torch.optim.Adam(D_params, lr=1e-4, betas=(0.9, 0.99))
    L1_loss = torch.nn.MSELoss()
    criterionGAN = GANLoss(device = device)

    total_epoch = cfg_pose.TRAIN.MAX_EPOCH
    i = 0
    r_G_GAN_loss,r_reconst_loss,r_KL_loss, r_D_GAN_loss = 0,0,0,0 
    for epoch in range(1,total_epoch):
        for data in train_data_loader:
            i += 1
            indiv_mels, ref_coeff, window_coeff_3dmm ,y  = data
            batch = {}
            first = ref_coeff[:,0,:].unsqueeze(1)
            batch['num_frames'] = 32  
            batch['indiv_mels'] = indiv_mels.unsqueeze(0)  
            batch['gt'] = torch.cat((first,window_coeff_3dmm), 1) [:,:,:70]   #gt的姿态
            batch['class']  = y

            output= audio2pose_model(batch)
            res_gt, res_pred = output['pose_motion_gt'], output['pose_motion_pred']
            mu, std =  output['mu'], output['logvar']
            
            #GAN loss
            G_GAN_loss = criterionGAN.compute_generator_loss(audio2pose_model.netD_motion, res_pred)
            reconst_loss = L1_loss(res_gt,res_pred)
            KL_loss = -0.5 * torch.mean(torch.sum(1 + std - mu.pow(2) - std.exp(), 1))

            total_G_loss = reconst_loss + KL_loss + G_GAN_loss * 0.7

            optimizer_G.zero_grad()
            total_G_loss.backward()
            optimizer_G.step()

            r_G_GAN_loss += G_GAN_loss
            r_reconst_loss += reconst_loss
            r_KL_loss += KL_loss

            #更新D
            if  i % 2 == 0:
                with torch.no_grad():  #生成器不更新
                    output = audio2pose_model(batch)
                res_gt, res_pred = output['pose_motion_gt'], output['pose_motion_pred']
                D_GAN_loss = criterionGAN.compute_discriminator_loss(audio2pose_model.netD_motion, res_gt, res_pred)
        
                total_D_loss = D_GAN_loss

                optimizer_D.zero_grad()
                total_D_loss.backward()
                optimizer_D.step()
            else:
                D_GAN_loss = 0

            r_D_GAN_loss += D_GAN_loss

            
            if i % args.interval == 0:
                r_G_GAN_loss,r_reconst_loss,r_KL_loss,r_D_GAN_loss = r_G_GAN_loss/args.interval, r_reconst_loss/args.interval,r_KL_loss/args.interval,r_D_GAN_loss/args.interval
                information = 'epoch: {}, iter: {}, G_GAN_loss: {:4f}, reconst_loss:{:4f}, KL_loss:{:4f}, D_GAN_loss:{:4f}'.format(epoch, i, r_G_GAN_loss, r_reconst_loss, r_KL_loss, r_D_GAN_loss)
                logger.info(information)
                r_G_GAN_loss,r_Greconst_loss,r_GKL_loss, r_D_GAN_loss = 0,0,0,0 
            
            if i % args.save_interval == 0:
                save_dict ={}
                save_ch_name = 'ep{}_iter{}.safetensors'.format(epoch,i)
                save_ch_path = os.path.join(worspace_path, save_ch_name)
                dicts = audio2pose_model.state_dict()
                new_dcits = {}
                for key in dicts:
                    newkey = 'audio2pose.' + key
                    new_dcits[newkey] = dicts[key]
                safetensors.torch.save_file(new_dcits, save_ch_path)  #保存








        