import logging
import os
import os.path as osp
import time
import pathlib
import logging.config
from color_formatter import ColoredFormatter  # 确保这个模块存在

# 定义未定义的变量
BASE_NAME = pathlib.Path(__file__).resolve().parent  # 假设 BASE_NAME 是当前脚本的目录
ENABLE_LOGGER = True  # 假设 ENABLE_LOGGER 是一个布尔值，用于控制日志记录的启用

BASE_LOG_DIR = os.path.join(BASE_NAME, "output/logs")
os.makedirs(BASE_LOG_DIR, exist_ok=True)

class CustomedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        if not ENABLE_LOGGER:
            return False
        filename = record.filename
        try:  # 将filename修改为相对路径，方便准确定位
            filename = str(
                pathlib.PurePath(record.pathname).relative_to(BASE_NAME)
            )
        except Exception as e:
            pass
        record.filename = filename
        return True

LOG_CFG = {
    "version": 1,
    # 禁用已经存在的logger实例
    "disable_existing_loggers": False,
    # 定义日志格式化工具
    "formatters": {
        "standard": {
            "format": "[%(asctime)s][%(name)s:%(filename)s:%(lineno)d][%(levelname)s]: %(message)s"
        },
        "simple": {
            "()": ColoredFormatter,  # 输出到终端所以带颜色
            "fmt": "%(log_color)s[%(asctime)s][%(filename)s:%(lineno)d]: %(message)s",
            "datefmt": "%H:%M:%S"
        },
    },
    # 过滤
    "filters": {
        "CustomedFilter": {"()": CustomedFilter, "name": "examTrueFilter"},
    },
    # 日志处理器
    "handlers": {
        "console": {
            "level": "INFO",
            "filters": ["CustomedFilter"],  # 只有在 ENABLE_LOGGER 为 True 时才在屏幕打印日志
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
        "eachLaunch": {  # 每次启动都会生成一个logger用来记录
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(
                BASE_LOG_DIR,
                f"hm_{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))}.log",
            ),  # 日志文件
            "maxBytes": 1024 * 1024 * 50,  # 日志大小 50M
            "backupCount": 3,
            "formatter": "standard",
            "encoding": "utf-8",
        },
        "default": {
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",  # 保存到文件，自动切
            "filename": os.path.join(BASE_LOG_DIR, "info.log"),  # 日志文件
            "maxBytes": 1024 * 1024 * 50,  # 日志大小 50M
            "backupCount": 3,
            "formatter": "standard",
            "encoding": "utf-8",
        },
        "error": {
            "level": "ERROR",
            "class": "logging.handlers.RotatingFileHandler",  # 保存到文件，自动切
            "filename": os.path.join(BASE_LOG_DIR, "err.log"),  # 日志文件
            "maxBytes": 1024 * 1024 * 50,  # 日志大小 50M
            "backupCount": 5,
            "formatter": "standard",
            "encoding": "utf-8",
        },
    },
    # loggers
    "loggers": {
        # 默认logger
        "hmquant": {
            "handlers": ["console", "default", "error", "eachLaunch"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOG_CFG)
logger = logging.getLogger("hmquant")
diff_logger = logger
hardware_graph_optimizer_logger = logging.getLogger("hmquant.hardware_graph_optimizer")
mix_quant_logger = logging.getLogger("hmquant.mix_quant")