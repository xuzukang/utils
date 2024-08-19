from datetime import datetime
import os
import sys

class Logger(object):
    def __init__(self, folder="logs"):
        # 获取当前时间并格式化为字符串
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 创建日志文件夹（如果不存在）
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # 定义日志文件名
        filename = os.path.join(folder, f"log_{current_time}.txt")
        
        # 打开日志文件
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()


sys.stdout = Logger(folder="logs")