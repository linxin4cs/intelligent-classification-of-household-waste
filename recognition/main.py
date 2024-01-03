import cv2
import torch
import UI
# import threading
import multiprocessing
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import serial
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import *
from PyQt5.Qt import QThread
# import matplotlib
# matplotlib.use("agg")
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
class gab_rec():
    def __init__(self):
        self.data = [0, 0, 0, 0]
        self.labels = ["有害垃圾", "可回收垃圾", "厨余垃圾", "其他垃圾"]
        """
        class myThread(threading.Thread):
            def __init__(self):
                threading.Thread.__init__(self)
                plt.ion()
                fig, ax = plt.subplots()
                plt.bar(range(len(data)), data, tick_label=labels)
                plt.show(block = False)
        """
        """
            一、有害垃圾：7
            二、可回收垃圾：5、6
            三、厨余垃圾：2、3、4
            四、其他垃圾：0、1
        """
        # self.idx_to_classes = {0: '其他垃圾_烟蒂', 1: '其他垃圾_破碎花盆及碟碗', 2: '厨余垃圾_水果果皮', 3: '厨余垃圾_水果果肉', 4: '厨余垃圾_菜叶菜根',
        #                   5: '可回收物_易拉罐', 6: '可回收物_饮料瓶', 7: '有害垃圾_干电池'}
        # self.ind_to_CLS = {7: 0, 5: 1, 6: 1, 2: 2, 3: 2, 4: 2, 0: 3, 1: 3}
        # self.model_ft = torch.load('./weights/resnet50_resize224_cls8_aug.pth', map_location="cpu")
        # self.model_ft.eval()
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.preprocess = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     self.normalize
        # ])

    def display_video(self):
        cap = cv2.VideoCapture("assets/Video.mp4")
        while (1):
            # get a frame
            ret, frame = cap.read()
            if frame is None:
                break
            # show a frame
            cv2.namedWindow("VIDEO", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("VIDEO", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("VIDEO", frame)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def wt(self,data):
        try:
          #端口，GNU / Linux上的/ dev / ttyUSB0 等 或 Windows上的 COM3 等
          portx="/dev/ttyUSB0"
          #波特率，标准值之一：50,75,110,134,150,200,300,600,1200,1800,2400,4800,9600,19200,38400,57600,115200
          bps=9600
          #超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）
          timex=5
          # 打开串口，并得到串口对象
          ser=serial.Serial(portx,bps,timeout=timex)

          # 写数据
          result=ser.write((str(data)+'\r\n').encode('utf-8'))
          print(data)

          ser.close()#关闭串口

        except Exception as e:
            print("---异常---：",e)

    def classify(self,img):
        outputs=self.model_ft(Variable(img))
        pred,ind=torch.max(F.softmax(outputs,dim=1).data,1)
        return pred.item(),ind.item()

class MyWindow(QMainWindow):
    def keyPressEvent(self, event):
        print("按下：" + str(event.key()))
        # 举例
        if(event.key() == Qt.Key_Escape):
            self.close()


if __name__=="__main__":
    grec=gab_rec()
    grec.display_video()
    app = QApplication(sys.argv)  # 创建应用程序
    mainwindow = MyWindow()  # 创建主窗口

    ui = UI.Ui_MainWindow()  # 调用first中的主窗口

    ui.setupUi(mainwindow)  # 向主窗口添加控件

    #mainwindow.show()  # 显示窗口
    mainwindow.showFullScreen()
    #ui.show_camera()
    # mp = multiprocessing.Process(target = grec.test(),args=())
    # mp.start()
    # tr = gabCls_Thread(ui)
    # tr.start()
    sys.exit(app.exec_())

