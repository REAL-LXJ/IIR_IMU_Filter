#coding:utf-8

import rosbag
import matplotlib.pyplot as plt
import numpy.fft as fft
import numpy as np
import scipy.fftpack as scifft
import scipy.signal as scisig
import math
import pywt

#@ rosbag解包获取imu数据
class ImuDataCreator():
    def _imu_data_export(self, rosbag_file, imu_topic, imu_path):
        imu_txt = open(imu_path, 'w')
        with rosbag.Bag(rosbag_file, 'r') as bag:   
            for topic,msg,t in bag.read_messages(): 
                if topic == imu_topic:
                    ax = "%.6f" % msg.linear_acceleration.x
                    ay = "%.6f" % msg.linear_acceleration.y
                    az = "%.6f" % msg.linear_acceleration.z
                    wx = "%.6f" % msg.angular_velocity.x
                    wy = "%.6f" % msg.angular_velocity.y
                    wz = "%.6f" % msg.angular_velocity.z
                    time = "%.6f" %  msg.header.stamp.to_sec()
                    imu_data = time + " " + ax + " " + ay + " " + az + " " + wx + " " + wy + " " + wz
                    # t_ax_data = time + " " + ax
                    # imu_txt.write(t_ax_data)
                    imu_txt.write(imu_data)
                    imu_txt.write('\n')
        imu_txt.close()

#@ 数据处理
class DataProcess():

    #% 200Hz-0.005s-5ms
    #@ 从txt文件里获取imu数据：t+ax+ay+az+wx+wy+wz
    def _get_imu_data(self, imu_txt_file):
        t0 = 1667463890.132030
        t = []
        ax = []
        ay = []
        az = []
        wx = []
        wy = []
        wz = []
        with open(imu_txt_file, 'r') as f: 
            for line in f:
                line = line.strip('\n').split(' ')
                t.append(float(line[0]) - t0)
                ax.append(float(line[1]))
                ay.append(float(line[2]))
                az.append(float(line[3]))
                wx.append(float(line[4]))
                wy.append(float(line[5]))
                wz.append(float(line[6]))
        return t, ax, ay, az, wx, wy, wz

    #@ 从txt文件里获取时间戳+单轴数据
    def _get_tax_data(self, ax_file):
        t0 = 1651203440.529707 #^ 首帧imu时间戳
        t = []
        ax = []
        with open(ax_file, 'r') as f:
            for line in f:
                line = line.strip('\n').split(' ')
                t.append(float(line[0]) - t0)
                ax.append(float(line[1]))
        return t, ax

    #@ 从txt文件里获取序列+单轴数据
    def _get_ax_data(self, ax_file):
        _s = 0
        s = []
        ax = []
        with open(ax_file, 'r') as f:
            for line in f:
                line = line.strip('\n').split(' ')
                ax.append(float(line[0]))
                s.append(_s)
                _s = _s + 1
        return s, ax

    #@ 向txt写入单轴原始加滤波数据
    def _write_ax_txt(self, ax, ax_filter, targetFile):
        target = open(targetFile, 'w')
        if (len(ax) == len(ax_filter)):
            for index1, _ax in enumerate(ax):
                for index2, _ax_filter in enumerate(ax_filter):
                    if index1 == index2:
                        #print(_t, _ax)
                        #target.writelines(str(_t) + ' ' + str(_ax) + "\n")             #* t + ax   
                        #target.writelines(str(_ax) + ' ' + str(_ax_filter) + "\n")     #* ax + ax_filter  训练数据集
                        target.writelines(str(_ax) + ' '+ str(0) + "\n")                #* ax + 0          测试数据集
        target.close()
        print("write txt sucessfully!")   

    #@ 绘制单轴原始数据+滤波数据图
    def _draw_ax(self, t, axo, axf):
        fig1, ax1 = plt.subplots()
        ax1.plot(t, axo, label='ax_orignal', ) 
        ax1.plot(t, axf, label='ax_filter', c='orangered')
        ax1.set_xlabel('timestamp') 
        ax1.set_ylabel('imu_ax') 
        ax1.set_title('OPPO_IMU') 
        ax1.legend()

    #@ 信号的傅里叶变换（原始信号和滤波信号的时频分析）
    def _fft(self, _t, _S0, _S1, _Fs):

        #* 导入原始数据
        t = _t
        S0 = _S0
        S1 = _S1

        #* 生成时间序列
        L = len(t)
        dt = 1/_Fs
        N = L
        Ls = L*dt
        t = np.arange(0, Ls, dt)

        #* fft傅里叶变换
        y0 = S0
        y1 = S1
        Y0 = fft.fft(y0)
        Y1 = fft.fft(y1)
        freqs = fft.fftfreq(L, t[1]-t[0])
        fs = freqs[freqs >= 0]
        M0 = abs(Y0[freqs >= 0])
        M1 = abs(Y1[freqs >= 0])
        A0 = M0/(N/2)
        A1 = M1/(N/2)

        #* 绘图
        plt.figure(figsize=(16,8))
        bias = 0.05#^ 滤波信号时偏

        #* 绘制时间-原始信号图
        plt.subplot(211)
        plt.plot(t, S0, label='ax')
        plt.plot(t-bias, S1, label='ax', c='orangered')
        plt.grid(linestyle=':')
        plt.xlabel('time(s)')
        plt.ylabel('value')
        plt.title('original signal')

        #* 绘制频率-幅值图
        plt.subplot(212)
        plt.plot(fs, A0)
        plt.plot(fs, A1, c='orangered')
        plt.ylim(0, 8)
        plt.grid(linestyle=':')
        plt.xlabel('frequency(Hz)')
        plt.ylabel('amplitude(V)')
        plt.title('Frequency-Amplitude Spectrum')

    #@ 信号的短时傅里叶变换（可以获取每个频率信号的时域信息）
    def _stft(self, _t, _S , _Fs):
        t = _t
        S = _S
        Fs = _Fs
        L = len(t)
        N = L
        win = 'hann'
        f, time, Z = scisig.stft(S, Fs, window=win, nperseg=256, noverlap=None, nfft=N, return_onesided=True)
        Z = np.abs(Z)
        plt.figure(figsize=(16,8))
        plt.pcolormesh(time, f, Z, vmin=0, vmax=Z.mean()*10)
        plt.title('STFT')
        plt.ylabel('frequency(Hz)')
        plt.xlabel('time')

    #@ 信号的小波变换
    def _cwt(self, _t, _S, _Fs):
        t = _t
        S = _S
        Fs = _Fs

        dt = 1/Fs

        wavename = "cgau8"
        totalscal = 256
        fc = pywt.central_frequency(wavename) #中心频率
        cparam = 2 * fc * totalscal
        scales = cparam/np.arange(totalscal,1,-1)
        [cwtmatr, frequencies] = pywt.cwt(S,scales,wavename,dt)#连续小波变换
        plt.figure(figsize=(16,8))
        plt.contourf(t, frequencies, abs(cwtmatr))
        plt.title('CWT')
        plt.ylabel('frequency(Hz)')
        plt.xlabel('time')

    #@ 信号分析
    def _analyze(self, _t, _S, _Fs, _start, _end, plot_analyse, plot_filter):
        
        #* 导入原始数据
        t = _t                                                                                                        #^ 时间序列
        S = _S                                                                                                        #^ 原始信号
        Fs = _Fs                                                                                                      #^ 采样频率(Hz)
        start = _start                                                                                                #^ 切片                                           
        end = _end                                                                                                    #^ 切片

        # Fs = 256
        # L = 1024

        #* 原始数据切片（局部信号分析）
        tslice = t[start:end]
        Sslice = S[start:end]                                                                                                                        

        #* 生成时间序列
        L = len(tslice)                                                                                                #^ 信号长度
        dt = 1/Fs                                                                                                      #^ 时间间隔(s)
        N = L                                                                                                          #^ 采样点数
        Lr = tslice[-1]-tslice[0]                                                                                      #^ 实际信号时长(s)
        Ls = L*dt                                                                                                      #^ 采样信号时长(s)
        tslice = np.arange(0, Ls, dt)                                                                                  #^ 时间序列时长

        # Sslice = 2+3*np.cos(2*np.pi*80*tslice-np.pi*150/180)+5*np.cos(2*np.pi*110*tslice+np.pi*125/180)+7*np.sin(2*np.pi*60*tslice+np.pi*250/180) +\
        #         + 2.5 *np.random.randn(len(tslice))
        
        print("序列采样点数 = ", N)
        print("序列采样频率 = ", Fs)
        print("序列时间间隔 = ", dt)
        print("实际信号时长 = ", Lr)
        print("采样信号时长 = ", Ls)
        
        #* fft频谱变换
        y = Sslice
        Yfft = fft.fft(y)                                                                                               #^ 对信号进行快速傅里叶变换
        Yfft_shift = scifft.fftshift(Yfft)                                                                              #& 对傅里叶变换进行shift排序
        freqs = fft.fftfreq(tslice.size, tslice[1]-tslice[0])                                                           #^ 双边频率
        freqs_shift = scifft.fftshift(freqs)                                                                            #& 对双边频率进行shift排序
        fs = freqs[freqs >= 0]                                                                                          #^ 单边频率
        M = abs(Yfft[freqs >= 0])                                                                                       #^ 复数的模(双边频谱)
        A = M/(N/2)                                                                                                     #^ 原信号单边频谱幅值
        A[0] = A[0]/2                                                                                                   #^ 直流分量
        E = M**2                                                                                                        #^ 能量
        E_shift = abs(Yfft_shift)**2                                                                                    #& 对能量进行shift排序
        P = E/len(E)                                                                                                    #^ 功率
        E_sum = np.sum(E)
        E_shift_sum = np.sum(E_shift)
        P_sum = np.sum(P)

        # print("信号能量 = ", E_sum)
        # print("信号功率 = ", P_sum)

        #*********************************************** 傅里叶模态分解 ***********************************************#
        #* 设置筛选条件，振幅阈值和能量阈值
        Amax = 0.8
        Emin = 1.0e6                                                                                              
        Emax = 1.75e8                                                                                                
        condition1 = A >= Amax
        condition2 = (E <= Emax)
        condition3 = (fs <= 10)
        id = np.where(condition3)                                                                                       #^ 获取信号索引

        #* 提取特定条件筛选后的信号特征
        fs_id = fs[id]                                                                                                  #^ 获取信号频率
        per_id = 1/fs_id[fs_id>0]                                                                                       #^ 获取信号周期
        A_id = A[id]                                                                                                    #^ 获取信号振幅
        phy_id = np.zeros((fs.size,),dtype=int)                                                                         #^ 初始化信号相位(弧度制)
        deg_id = np.zeros((fs.size,),dtype=int)                                                                         #^ 初始化信号角度(角度制)
        ReS_id = np.zeros(tslice.size)                                                                                  #! 初始化信号成分

        for i in range(0, A_id.size):
            for j in range(0, A.size):
                if A_id[i] == A[j]:
                    phy_id[j] = np.angle(Yfft[j], deg=0)                                                                #^ 获取信号相位
                    deg_id[j] = np.angle(Yfft[j], deg=1)                                                                #^ 获取信号角度

        #* 傅里叶模态分解
        print("信号数目 = ", fs_id.size, len(per_id), len(A_id), len(deg_id[id]))                                        #^ 信号数量
        print("信号频率 = ", fs_id)                                                                                      #^ 信号频率
        print("信号周期 = ", per_id)                                                                                     #^ 信号周期
        print("信号振幅 = ", A_id)                                                                                       #^ 信号振幅
        print("信号相位 = ", deg_id[id])                                                                                 #^ 信号初始相位(角度制)

        #* 构造信号 S = A0 + A*cos(2*pi*fs*t+pi/180*θ)
        for i in range(0, fs_id.size):
            print('{0:2}\t{1:5}\t{2:.5f}\t{3:5}\t'.format(i, fs_id[i], A_id[i], deg_id[id][i]))
            ReS_id += A_id[i]*np.cos(2*np.pi*fs_id[i]*tslice+np.pi*deg_id[id][i]/180)                                   #^ 构造信号
            # S_cid.append(S_id)
        # print("S_id = ", ReS_id)
        Err = y - ReS_id                                                                                                #^ 误差
        
        mse = np.sum(Err**2)/len(Err)                                                                                   #^ 均方误差
        mean = np.mean(Err)                                                                                             #^ 均值误差
        var = np.var(Err)                                                                                               #^ 方差
        std = np.std(Err)                                                                                               #^ 标准差
        #*********************************************** 傅里叶模态分解 ***********************************************#

        #************************************************** 滤波方法 *************************************************#
        #* 1.利用fft和ifft进行滤波
        #* 2.利用butter和filtfilt进行滤波
        #* 设置筛选条件去噪，噪声特点：高频，低能量
        #* 1.利用频率大小，选取不同频带(frequency band)和傅里叶变换保留成分，达到去除噪声的目的
        #* 2.利用能量强弱
        #* 3.带阻、陷波
        freq_band = abs(freqs_shift) 
        energy = E_shift
        filter1 = (freq_band <= 10) & (energy > 0.5e7)
        filter2 = energy > 0.5e7
        filter3 = freq_band <= 10

        E_shift_filter1 = E_shift*filter1
        Y_shift_filter1 = Yfft_shift*filter1
        _Y_shift_filter1 = scifft.fftshift(Y_shift_filter1)
        Y_filter1 = scifft.ifft(_Y_shift_filter1)

        E_shift_filter2 = E_shift*filter2
        Y_shift_filter2 = Yfft_shift*filter2
        _Y_shift_filter2 = scifft.fftshift(Y_shift_filter2)
        Y_filter2 = scifft.ifft(_Y_shift_filter2)
        
        E_shift_filter3 = E_shift*filter3
        Y_shift_filter3 = Yfft_shift*filter3
        _Y_shift_filter3 = scifft.fftshift(Y_shift_filter3)
        Y_filter3 = scifft.ifft(_Y_shift_filter3)

        #* 降噪相关指标(能量比+相似度)，降噪效果与降噪能量比成正比，与降噪相似度成反比
        Efilter1 = np.sum(abs(Y_filter1)**2)/E_shift_sum
        Efilter2 = np.sum(abs(Y_filter2)**2)/E_shift_sum
        Efilter3 = np.sum(abs(Y_filter3)**2)/E_shift_sum
        E_rate = (Efilter1, Efilter2, Efilter3)

        Sfilter1 = np.sum(abs(y-Y_filter1)**2)/len(y)
        Sfilter2 = np.sum(abs(y-Y_filter2)**2)/len(y)
        Sfilter3 = np.sum(abs(y-Y_filter3)**2)/len(y)
        S_deg = (1/Sfilter1, 1/Sfilter2, 1/Sfilter3)

        #* 建立优良频率降噪模型及算法
        alpha = 0.2
        E_rate = E_rate/np.max(E_rate)
        S_deg = S_deg/np.max(S_deg)
        Sr_deg = S_deg[::-1]
        model1 = alpha*E_rate[0]+(1-alpha)*S_deg[0]
        model2 = alpha*E_rate[1]+(1-alpha)*S_deg[1]
        model3 = alpha*E_rate[2]+(1-alpha)*S_deg[2]
        model = (model1, model2, model3)

        wn = 2*10/200
        b,a = scisig.butter(N=8, Wn=wn, btype='lowpass')
        filterdata = scisig.filtfilt(b, a ,y)
        #************************************************** 滤波方法 *************************************************#
            
        if plot_analyse:
            #* 画图
            plt.figure(figsize=(16,8))
            #% 原信号
            plt.subplot(311)
            plt.plot(t, S, label='ax')
            # plt.legend();
            plt.grid(linestyle=':')
            plt.xlabel('time(s)')
            plt.ylabel('value')
            plt.title('original signal')

            #% 原信号取样
            # plt.subplot(221)
            # plt.plot(tslice, Sslice, c='orangered', label='ax_slice')
            # # plt.legend();
            # plt.grid(linestyle=':')
            # plt.xlabel('time(s)')
            # plt.ylabel('value')
            # plt.title('original signal slice')

            #% 频率幅值谱
            plt.subplot(312)
            plt.plot(fs, A, c='orangered')
            plt.ylim(0, 8)
            # plt.legend()
            plt.grid(linestyle=':')
            plt.xlabel('frequency(Hz)')
            plt.ylabel('amplitude(V)')
            plt.title('Frequency-Amplitude Spectrum')

            #% 频率相位谱(弧度制)
            # plt.subplot(234)
            # plt.plot(fs, phi, c='orangered')
            # # plt.legend()
            # plt.grid(linestyle=':')
            # plt.xlabel('frequency(Hz)')
            # plt.ylabel('phase(rad)')
            # plt.title('Frequency-Phase Spectrum')
            
            #% 频率相位谱(角度制)
            # plt.subplot(414)
            # plt.plot(fs, deg_id, c='orangered')
            # # plt.legend()
            # plt.grid(linestyle=':')
            # plt.xlabel('frequency(Hz)')
            # plt.ylabel('phase(degree)')
            # plt.title('Frequency-Phase Spectrum')

            #% 频率能量谱
            plt.subplot(313)
            plt.plot(fs, E, c='orangered')
            plt.ylim(0, 2e8)
            # plt.legend()
            plt.grid(linestyle=':')
            plt.xlabel('frequency(Hz)')
            plt.ylabel('energy(V*2)')
            plt.title('Frequency-Energy Spectrum')

            #% 频率功率谱
            # plt.subplot(246)
            # plt.plot(fs, P, c='orangered')
            # plt.ylim(0, 5e4)
            # # plt.legend()
            # plt.grid(linestyle=':')
            # plt.xlabel('frequency(Hz)')
            # plt.ylabel('power(V*2)')
            # plt.title('Frequency-Power Spectrum')

            #% 构造信号与原始信号对比图
            # plt.subplot(514)
            # plt.plot(tslice, Sslice, label='orignal_signal')
            # plt.plot(tslice, ReS_id, c='orangered', label='construct_signal')
            # # plt.ylim(0, 3)
            # # plt.legend()
            # plt.grid(linestyle=':')
            # plt.xlabel('time(s)')
            # plt.ylabel('value')
            # plt.title('Original_construct_signal_compare')

            #% 构造信号与原始信号误差
            # plt.subplot(515)
            # plt.plot(tslice, Err, c='green', label='signal_error')
            # # plt.ylim(0, 3)
            # # plt.legend()
            # plt.grid(linestyle=':')
            # plt.xlabel('time(s)')
            # plt.ylabel('err')
            # plt.title('Original_construct_signal_error')

            plt.tight_layout()

        if plot_filter:
            
            plt.figure(figsize=(16,8))

            #% 平移频率幅值谱
            # plt.subplot(513)
            # plt.plot(freqs_shift, Yfft_shift)
            # plt.plot(freqs_shift, Y_shift_filter1, c='orangered')
            # plt.plot(freqs_shift, Y_shift_filter2, c='green')
            # plt.ylim(0, 10000)
            # # plt.legend()
            # plt.grid(linestyle=':')
            # plt.xlabel('frequency(Hz)')
            # plt.ylabel('amplitude(V)')
            # plt.title('Frequency-Amplitude Spectrum shift')

            #% 平移频率能量谱
            # plt.subplot(512)
            # plt.plot(freqs_shift, E_shift, c='orangered')
            # # plt.plot(freqs_shift, Y_shift_filter1, c='orangered')
            # plt.ylim(0, 1e8)
            # # plt.legend()
            # plt.grid(linestyle=':')
            # plt.xlabel('frequency(Hz)')
            # plt.ylabel('energy(V*2)')
            # plt.title('Frequency-Energy Spectrum shift')

            plt.subplot(511)
            plt.plot(tslice, Sslice, label='orignal_signal')
            plt.plot(tslice, np.real(Y_filter1), c='orangered', label='fb1')
            # plt.ylim(0, 3)
            # plt.legend()
            plt.grid(linestyle=':')
            plt.xlabel('time(s)')
            plt.ylabel('value')
            plt.title('pass-filter-way1')

            plt.subplot(512)
            plt.plot(tslice, Sslice, label='orignal_signal')
            plt.plot(tslice, np.real(Y_filter2), c='orangered', label='fb2')
            # plt.ylim(0, 3)
            # plt.legend()
            plt.grid(linestyle=':')
            plt.xlabel('time(s)')
            plt.ylabel('value')
            plt.title('pass-filter-way2')

            plt.subplot(513)
            plt.plot(tslice, Sslice, label='orignal_signal')
            plt.plot(tslice, np.real(Y_filter3), c='orangered', label='fb3')
            plt.plot(tslice, filterdata, c='green', label='fb3')
            # plt.ylim(0, 3)
            # plt.legend()
            plt.grid(linestyle=':')
            plt.xlabel('time(s)')
            plt.ylabel('value')
            plt.title('pass-filter-way3')

            #% 降噪能量比
            # plt.subplot(514)
            # plt.plot(E_rate, c='red', linewidth=1.0, linestyle ='--')
            # # plt.ylim(0, 3)
            # # plt.legend()
            # plt.grid(linestyle=':')
            # plt.xticks([0,1,2],['way1','way2','way3'])
            # plt.xlabel('filter algorithm')
            # plt.ylabel('energy_rate')
            # plt.title('Noise reduction energy rate')

            #% 降噪相似度
            # plt.subplot(515)
            # plt.plot(S_deg, c='red', linewidth=1.0, linestyle ='--')
            # # plt.ylim(0, 3)
            # # plt.legend()
            # plt.grid(linestyle=':')
            # plt.xticks([0,1,2],['way1','way2','way3'])
            # plt.xlabel('filter algorithm')
            # plt.ylabel('similarity')
            # plt.title('Noise reduction similarity')
            
            #% 优良降噪模型
            # plt.subplot(514)
            # plt.plot(E_rate, 'b', linewidth=1.0, label = 'energy_rate')
            # plt.plot(S_deg, 'c', linewidth=1.0, label = 'similarity')
            # plt.plot(model, 'r', linewidth=1.0, label = 'goal function')
            # # plt.ylim(0, 3)
            # plt.legend()
            # plt.grid(linestyle=':')
            # plt.xticks([0,1,2],['way1','way2','way3'])
            # plt.xlabel('filter algorithm')
            # plt.ylabel('goal function')
            # plt.title('goal function & algorithm target')

            #% 构造信号与原始信号对比图
            plt.subplot(514)
            plt.plot(tslice, Sslice, label='orignal_signal')
            plt.plot(tslice, ReS_id, c='orangered', label='construct_signal')
            # plt.ylim(0, 3)
            # plt.legend()
            plt.grid(linestyle=':')
            plt.xlabel('time(s)')
            plt.ylabel('value')
            plt.title('Original_construct_signal_compare')

            #% 构造信号与原始信号误差
            plt.subplot(515)
            plt.plot(tslice, Err, c='green', label='signal_error')
            # plt.ylim(0, 3)
            # plt.legend()
            plt.grid(linestyle=':')
            plt.xlabel('time(s)')
            plt.ylabel('err')
            plt.title('Original_construct_signal_error')

            plt.tight_layout()

#* 200Hz-0.005s-5ms
if __name__ == '__main__':

    #* 读入相关路径
    #rosbag_file = "/home/lxj/HDD/Dataset/oppo/D435i/2022-07-28-11-35-43_zhexian.bag"
    read_rosbag_file = "/home/lxj/Dataset/2022-02-17-21-05-41.bag"
    read_imu_path1 = "/home/lxj/桌面/oppo_imu_filter/dogcar_imu_dog_001.txt"
    read_imu_path2 = "/home/lxj/桌面/oppo_imu_filter/dogcar_imu_car_001.txt"
    read_imu_topic = "/cam_3/imu" 
    ax_filter_file = "/home/lxj/桌面/pytorch_imu_filter/MLP/S_fast_ax_ax_filter.txt"

    #* 手持imu分析
    handhold_move_rosbag_file = "/home/lxj/Dataset/test_imu_524_handhold_move/handhold_move_201.bag"
    handhold_move_imu_path = "/home/lxj/桌面/oppo_imu_filter/handhold_move_001.txt"
    handhold_move_imu_topic = "/cam_1/imu"

    #* 机器狗imu [march, straight, circle, head, body]
    dog_imu_file = "/home/lxj/HDD/Dataset/oppo/D435i/2022-08-03-17-16-08_N_fast.bag"
    dog_imu_topic1 = "/master/imu"
    dog_imu_topic2 = "/imu2"
    dog_imu_path = "/home/lxj/桌面/oppo_imu_filter/S_slow_master_imu.txt"
    test_path = "/home/lxj/桌面/oppo_imu_filter/imu_test.txt"

    march_head_dog_imu_path = "/home/lxj/桌面/oppo_imu_filter/dog_imu/march_head_dog_imu.txt"
    straight_head_dog_imu_path = "/home/lxj/桌面/oppo_imu_filter/dog_imu/straight_head_dog_imu.txt"
    circle_head_dog_imu_path = "/home/lxj/桌面/oppo_imu_filter/dog_imu/circle_head_dog_imu.txt"
    dynamic_head_dog_imu_path = "/home/lxj/桌面/oppo_imu_filter/dynamic_imu.txt"

    ax_orig_file = "/home/lxj/桌面/oppo_imu_filter/dynamic_ax.txt"
    ax_proc_file = "/home/lxj/桌面/oppo_imu_filter/dynamic_ax_filter.txt"
    
    #* imu解包
    # creator = ImuDataCreator()
    # creator._imu_data_export(dog_imu_file, dog_imu_topic1, dog_imu_path)
    # print("complete!")

    datapro = DataProcess()

    #* 解析数据
    #* [march, straight, circle, head, body]
    dog = datapro._get_imu_data(dog_imu_path)                     
    # dog = datapro._get_imu_data(read_imu_path1)
    # car = datapro._get_imu_data(read_imu_path2)
    mh = datapro._get_imu_data(march_head_dog_imu_path)
    sh = datapro._get_imu_data(straight_head_dog_imu_path)
    ch = datapro._get_imu_data(circle_head_dog_imu_path)


    _start = 0
    _end = len(dog[0])
    _end1 = len(mh[0])
    _end2 = len(sh[0])
    _end3 = len(ch[0])
    _Fs = 200

    #* 频域分析
    #* fig1-6 --- ax,ay,az,gx,gy,gz
    # datapro._analyze(mh[0], mh[1], _Fs, _start, _end1, plot_analyse=1, plot_filter=0)
    # datapro._analyze(sh[0], sh[1], _Fs, _start, _end2, plot_analyse=1, plot_filter=0)
    # datapro._analyze(ch[0], ch[1], _Fs, _start, _end3, plot_analyse=1, plot_filter=0)
    # datapro._analyze(dog[0], dog[1], _Fs, _start, _end, plot_analyse=1, plot_filter=0)
    # datapro._analyze(dog[0], dog[2], _Fs, _start, _end, plot_analyse=1, plot_filter=0)
    # datapro._analyze(dog[0], dog[3], _Fs, _start, _end, plot_analyse=1, plot_filter=0)
    # datapro._analyze(dog[0], dog[4], _Fs, _start, _end, plot_analyse=1, plot_filter=0)
    # datapro._analyze(dog[0], dog[5], _Fs, _start, _end, plot_analyse=1, plot_filter=0)
    # datapro._analyze(dog[0], dog[6], _Fs, _start, _end, plot_analyse=1, plot_filter=0)

    # datapro._analyze(car[0], car[1], _Fs, _start, _end, plot_analyse=1, plot_filter=0)
    # datapro._analyze(car[0], car[2], _Fs, _start, _end, plot_analyse=1, plot_filter=0)
    # datapro._analyze(car[0], car[3], _Fs, _start, _end, plot_analyse=1, plot_filter=0)
    # datapro._analyze(car[0], car[4], _Fs, _start, _end, plot_analyse=1, plot_filter=0)
    # datapro._analyze(car[0], car[5], _Fs, _start, _end, plot_analyse=1, plot_filter=0)
    # datapro._analyze(car[0], car[6], _Fs, _start, _end, plot_analyse=1, plot_filter=0)

    t0, ax0 = datapro._get_ax_data(ax_orig_file)
    t1, ax1 = datapro._get_ax_data(ax_proc_file)
    datapro._fft(t0, ax0, ax1, 200)
    datapro._stft(t0, ax0, 200)
    datapro._cwt(t0, ax0, 200)
    plt.show()
