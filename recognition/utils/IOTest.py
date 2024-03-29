import serial

def wt(data):
    try:
        # 端口，GNU / Linux上的/ dev / ttyUSB0 等 或 Windows上的 COM3 等
        portx = "/dev/ttyUSB0"
        # 波特率，标准值之一：50,75,110,134,150,200,300,600,1200,1800,2400,4800,9600,19200,38400,57600,115200
        bps = 9600
        # 超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）
        timex = 5
        # 打开串口，并得到串口对象
        ser = serial.Serial(portx, bps, timeout=timex)

        # 写数据
        result = ser.write((str(data) + '\r\n').encode('utf-8'))
        print(data)

        ser.close()  # 关闭串口

    except Exception as e:
        print("---异常---：", e)


def receive():
    portx = "/dev/ttyUSB0"
    # 波特率，标准值之一：50,75,110,134,150,200,300,600,1200,1800,2400,4800,9600,19200,38400,57600,115200
    bps = 9600
    # 打开串口，并得到串口对象
    ser = serial.Serial(portx, bps, timeout=None)
    while True:
        try:
            # 读数据
            result = ser.read(1)
            print("result",result)
            if result == "6":
                ser.close()  # 关闭串口
                break
        except Exception as e:
            print("---异常---：", e)

data = 6
wt(data)
receive()
