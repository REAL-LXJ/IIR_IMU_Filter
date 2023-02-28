#coding:utf-8
import rospy
import math
from sensor_msgs.msg import Imu

import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph as pg
import numpy as np
import message_filters


At1, At2 = [], []
Ax1, Ay1, Az1 = [], [], []
Ax2, Ay2, Az2 = [], [], []
last_t1 = 0
last_t2 = 0
bias = 0

#* GUI界面
class ImuGui(QWidget):

    def __init__(self):       
        super(ImuGui, self).__init__()
        self.initUI()
        
    def initUI(self):
        self.setGeometry(0, 0, 1200, 800)                                     # 设置GUI界面的大小
        self.setWindowTitle('imu_filter')                                     # 界面窗口名称
        layout_chart = QtWidgets.QGridLayout()                                # 表格布局
        self.setLayout(layout_chart)
        self.pw = pg.PlotWidget(name="time")
        self.pw.showGrid(x=True, y=True)                                      # 绘图控件显示网格
        self.plt_ax1 = self.pw.plot(At1, Ax1, pen=(0,255,0),clear=True)       # 画笔颜色设置
        self.plt_ay1 = self.pw.plot(At2, Ay1, pen=(255,0,0))
        self.plt_az1 = self.pw.plot(At1, Az1, pen=(0,0,255))
        self.plt_ax2 = self.pw.plot(At2, Ax2, pen=(255,128,0))
        self.plt_ay2 = self.pw.plot(At1, Ay2, pen=(160,32,240))
        self.plt_az2 = self.pw.plot(At2, Az2, pen=(255,255,0))
        layout_chart.addWidget(self.pw, 0, 0, 9, 10)
        
        #* 按键+文本框
        self.button = QPushButton("plot")
        layout_chart.addWidget(self.button, 10, 0, 1, 1)
        self.text_edit = QLineEdit("test")
        layout_chart.addWidget(self.text_edit, 10, 1, 1, 2)

    #* 绘制单个imu信号    
    def draw_signal(self, t, ax1, ay1, az1):
        At1.append(t)
        Ax1.append(ax1)  
        Ay1.append(ay1)
        Az1.append(az1) 
        self.plt_ax1.setData(At1, Ax1)                        
        # self.ay_curve.setData(At, Ay)
        # self.az_curve.setData(At, Az)

    #* 绘制两个imu信号:原始信号+滤波信号    
    def draw_signals(self, t1, ax1, ay1, az1, ax2, ay2, az2):
        At1.append(t1)
        At2.append(t1 - bias)
        Ax1.append(ax1)  
        Ay1.append(ay1)
        Az1.append(az1) 
        Ax2.append(ax2)
        Ay2.append(ay2)
        Az2.append(az2)
        self.plt_ax1.setData(At1, Ax1)
        self.plt_ax2.setData(At2, Ax2)                          
        self.plt_ay1.setData(At1, Ay1)
        self.plt_ay2.setData(At2, Ay2)
        self.plt_az1.setData(At1, Az1)
        self.plt_az2.setData(At2, Az2)

def imu_callback0(msg1, gui):
    global last_t1
    t1 = msg1.header.stamp.to_sec()
    dt = t1 - last_t1
    last_t1 = t1
    hz = 1./dt
    ax1 = msg1.linear_acceleration.x
    ay1 = msg1.linear_acceleration.y
    az1 = msg1.linear_acceleration.z
    gx1 = msg1.angular_velocity.x
    gy1 = msg1.angular_velocity.y
    gz1 = msg1.angular_velocity.z

    print('Got IMU at {} s ({:.0f} Hz): {:.2f}, {:.2f}, {:.2f}, \t {:.2f}, {:.2f}, {:.2f}'
            .format(t1, hz, ax1, ay1, ax1, gx1, gy1, gz1))
            
    gui.draw_signal(t1, ax1, ay1, az1)

def imu_callback1(msg1, msg2, gui):
    global last_t1, last_t2
    t1 = msg1.header.stamp.to_sec()
    # t2 = msg2.header.stamp.to_sec()
    dt1 = t1 - last_t1
    # dt2 = t2 - last_t2
    last_t1 = t1
    # last_t2 = t2
    hz1 = 1./dt1
    # hz2 = 1./dt2
    ax1 = msg1.linear_acceleration.x
    ay1 = msg1.linear_acceleration.y
    az1 = msg1.linear_acceleration.z
    # gx1 = msg1.angular_velocity.x
    # gy1 = msg1.angular_velocity.y
    # gz1 = msg1.angular_velocity.z
    ax2 = msg2.linear_acceleration.x
    ay2 = msg2.linear_acceleration.y
    az2 = msg2.linear_acceleration.z

    # print('Got IMU at {} s ({:.0f} Hz): {:.2f}, {:.2f}'.format(t, hz, ax1, ax2))
    print('Got IMU at {} s ({:.0f} Hz): {:.2f}, {:.2f}, {:.2f}, \t {:.2f}, {:.2f}, {:.2f}'
            .format(t1, hz1, ax1, ax2, ay1, ay2, az1, az2))
    gui.draw_2signal(t1, ax1, ay1, az1, ax2, ay2, az2)

def main():

    app = QtWidgets.QApplication(sys.argv)
    gui = ImuGui()
    gui.show()

    #* 机器狗-小车数据集
    imu_dog_topic = "/cam_2/imu"
    imu_car_topic = "/cam_3/imu"

    #* 机器狗head-body数据集
    imu_head_topic = "/imu"
    imu_body_topic = "/imu2"

    imu_filter_topic = "/master/filter_imu"
    
    rospy.init_node("imu_analysis",anonymous=True)

    #* 原始数据
    # rospy.Subscriber(imu_head_topic, Imu, imu_callback, gui)
    # rospy.Subscriber(imu_filter_topic, Imu, imu_callback, gui)

    #* 原始数据+滤波数据
    orig_imu_sub = message_filters.Subscriber(imu_head_topic, Imu)
    filt_imu_sub = message_filters.Subscriber(imu_filter_topic, Imu)
    ts = message_filters.TimeSynchronizer([orig_imu_sub, filt_imu_sub], 10)
    ts.registerCallback(imu_callback1, gui)

    print("waiting imu data...")

    sys.exit(app.exec_())
    rospy.spin()

if __name__ =='__main__':
    main()