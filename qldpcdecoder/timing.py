import time
from functools import wraps
def timing(decoder_info=None, log_file=None):
    """
    带参数的装饰器函数，支持输出到日志文件。
    :param decoder_info: 解码器的描述信息（可选）
    :param log_file: 日志文件路径（可选）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time

            message = f"[{decoder_info}] Function '{func.__name__}' took {elapsed_time:.4f} seconds to run."
            if log_file:
                with open(log_file, "a") as f:
                    f.write(message + "\n")
            else:
                print(message)

            return result
        return wrapper
    return decorator