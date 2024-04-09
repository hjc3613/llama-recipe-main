
import os
from datetime import datetime

import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logger_v1():
    # 日志文件命名
    log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    log_filename = "output_" + datetime.now().strftime("%Y-%m-%d.log")
    log_path = os.path.join(log_folder, log_filename)
    # 检查日志文件夹是否存在，如果不存在则创建
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)


    # 配置 logger
    logger = logging.getLogger("mylogger")
    logger.setLevel(logging.DEBUG)

    # 设定info级别的日志文件和格式
    info_file_handler = TimedRotatingFileHandler(os.path.join(log_folder, "info.log"), when="midnight", interval=1, backupCount=7)
    info_file_handler.setLevel(logging.INFO)
    info_file_formatter = logging.Formatter('%(asctime)s - %(process)d - %(lineno)d - [INFO] - %(message)s')
    info_file_handler.setFormatter(info_file_formatter)
    logger.addHandler(info_file_handler)

    # 设定error级别的日志文件和格式
    error_file_handler = TimedRotatingFileHandler(os.path.join(log_folder, "error.log"), when="midnight", interval=1, backupCount=7)
    error_file_handler.setLevel(logging.ERROR)
    error_file_formatter = logging.Formatter('%(asctime)s - %(process)d - %(lineno)d - [ERROR] - %(message)s')
    error_file_handler.setFormatter(error_file_formatter)
    logger.addHandler(error_file_handler)

    return logger


def setup_logger():
    # 日志文件命名
    log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    log_filename = "output_" + datetime.now().strftime("%Y-%m-%d.log")
    log_path = os.path.join(log_folder, log_filename)
    # 检查日志文件夹是否存在，如果不存在则创建
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    # 配置日志记录器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建日志记录器
    # logger = logging.getLogger('NLP')
    logger = logging.getLogger(__name__)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 设置处理器的格式器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger_v1()


if __name__ == '__main__':

    # 创建全局的日志记录器
    # logger = setup_logger()
    logger = setup_logger_v1()
    # 使用方式
    logger.info('This is a info message')
    logger.error('This is an error message')