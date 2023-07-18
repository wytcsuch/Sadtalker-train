# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)  #创建一个记录器
    logger.setLevel(logging.DEBUG)  #只收集debug之上的日志 日志等级划分：NOTEST(0) DEBUG(10) INFO(20) WARNING(30) ERROR(40) CRITICAL(50)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process  主进程在屏幕上输出
    #将日志记录（log record）发送到合适的目的地（destination），比如文件，socket等。
    # 一个logger对象可以通过addHandler方法添加0到多个handler，每个handler又可以定义不同日志级别，以实现日志分级过滤显示
    #下面就是创建了两个handler,当dist_rank==0时,输出信息到终端,其余直接输出的文件中
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)  #能够将日志信息输出到sys.stdout
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)  #将console_handler添加到所创建的记录器中:每次进行记录，console都会输出

    # create file handlers  每一个进程都写入到独立的文件中
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')  #能够将日志信息写入文件
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)   #将file_handler添加到所创建的记录器中:每次进行记录，都会写入文件中

    return logger
