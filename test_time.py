import time
import datetime

try:
    while True:
        # 获取当前时间，精确到微秒
        current_time = datetime.datetime.now()
        
        # 格式化时间，包含毫秒的千分位
        # %H:%M:%S 为时:分:秒
        # %f 为微秒（6位数），取前3位即为毫秒
        formatted_time = current_time.strftime("%H:%M:%S.%f")[:-3]
        
        # 转换为字符串并打印
        time_str = str(formatted_time)
        print(time_str)
        
        # 短暂暂停，避免输出过快
        time.sleep(0.005)
        
except KeyboardInterrupt:
    # 处理Ctrl+C退出
    print("\n程序已退出")
