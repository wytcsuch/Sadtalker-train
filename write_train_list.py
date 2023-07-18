import os
from tqdm import tqdm
import random
import shutil
from PIL import Image
from src.utils.get_file import Get_img_dirs

def write_txt(dirs, txtdir):  
    with open(txtdir, 'wt') as f_out:
        for i in tqdm(range(len(dirs))):
            content = dirs[i] + ' ' + str(i)
            f_out.write('{:s}\r\n'.format(content))


if __name__ == '__main__':
    data_dir = '/metahuman/wyt/debug/images'
    txt_dir = '/metahuman/wyt/debug/images.txt'
    data_dirs = Get_img_dirs(data_dir)
    data_dirs.sort()
    write_txt(data_dirs, txt_dir)

        