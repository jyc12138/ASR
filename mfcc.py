from pydub import AudioSegment
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
import wave
import os
from python_speech_features import *
from scipy import io


def get_ms_part_wav(main_wav_path, start_time, end_time, part_wav_path):
    start_time = int(start_time)
    end_time = int(end_time)
    sound = AudioSegment.from_wav(main_wav_path)
    output = sound[start_time: end_time]
    output.export(part_wav_path, format="wav")

def sgn(data):
    if data >= 0:
        return 1
    else:
        return 0

def calculateEnergy(wave_data):
    energy = []
    sum = 0
    for i in range(len(wave_data)):
        sum += (wave_data[i] * wave_data[i])
        # print(sum)
        if (i + 1) % 256 == 0:
            energy.append(sum)
            sum = 0
        elif i == len(wave_data) - 1:
            energy.append(sum)
    # print(energy,"energy")
    return energy

def calculateZCR(wave_data):
    zcr = []
    sum = 0
    for i in range(len(wave_data)):
        if i % 256 == 0:
            continue
        sum += np.abs(sgn(wave_data[i]) - sgn(wave_data[i - 1]))
        if (i + 1) % 256 == 0:
            zcr.append(float(sum / 255))
            sum = 0
        elif i == len(wave_data) - 1:
            zcr.append(float(sum / 255))
    # print(zcr,"zcr")
    return zcr

def endPointDetect(wave_data, energy, zcr):
    sum = 0
    avgEnergy = 0
    for e in energy:
        sum += e
    avgEnergy = sum / len(energy)

    sum = 0
    for e in energy[:5]:
        sum += e
    ML = sum / 5
    MH = avgEnergy / 2
    ML = (ML + MH) / 2

    sum = 0
    for z in zcr[:5]:
        sum += float(sum + z)
    Zs = sum / 5

    A = []
    B = []
    C = []

    flag = 0
    for i in range(len(energy)):
        if len(A) == 0 and flag == 0 and energy[i] > MH:
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 20 > A[len(A) - 1]:
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 20 <= A[len(A) - 1]:
            A = A[:len(A) - 1]
            flag = 1

        if flag == 1 and energy[i] < MH:
            A.append(i)
            flag = 0

    for j in range(len(A)):
        i = A[j]
        if j % 2 == 1:
            while i < len(energy) and energy[i] > ML:
                i += 1
            B.append(i)
        else:
            while i > 0 and energy[i] > ML:
                i -= 1
            B.append(i)

    for j in range(len(B)):
        i = B[j]
        if j % 2 == 1:
            while i < len(zcr) and zcr[i] >= 3 * Zs:
                i += 1
            C.append(i)
        else:
            while i > 0 and zcr[i] >= 3 * Zs:
                i -= 1
            C.append(i)

    # print(A,"A")
    # print(B,"B")
    # print(C,"C")
    return C


def validation(wav_path):
    # 语音有效段检测-短时平均过零率
    f = wave.open(wav_path, "rb")
    params = f.getparams()
    # 读取格式信息
    # (声道数、量化位数、采样频率、采样点数、压缩类型、压缩类型的描述)
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()

    # 将字符串转换为数组，得到一维的short类型的数组
    wave_data = np.fromstring(str_data, dtype=np.short)
    # 赋值的归一化
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))
    # 整合左声道和右声道的数据
    wave_data = np.reshape(wave_data, [nframes, nchannels])
    # wave_data.shape = (-1, 2)
    # print(wave_data)
    energy = calculateEnergy(wave_data)
    zcr = calculateZCR(wave_data)
    N = endPointDetect(wave_data, energy, zcr)
    i = 0
    while i < len(N):
        N[i] = N[i] * 256
        i += 1
    fft_signal = np.fft.fft(wave_data)  # 语音信号FFT变换
    fft_signal = abs(fft_signal)  # 取变换结果的模
    # plt.figure(figsize=(10, 4))
    time = np.arange(0, nframes) * framerate / nframes
    #plt.plot(time, fft_signal, c="g")
    #plt.grid()
    #plt.show()
    return N


def get_mfcc(wav_path):
    rate, audio = wav.read(wav_path)
    return mfcc(audio, rate)

def getFeatures(wav_path):
    N = validation(wav_path)
    N = sorted(N)
    # print(wav_path.split('/')[-1])
    if N[0]>5000:
        N[0] = 5000
    print(N, "N")
    get_ms_part_wav(wav_path, N[0], N[-1], '/..{}'.format(wav_path.split('/')[-1]))
    #orig1 = get_mfcc('E:/学习/中科院科研/AudioRecognition-master/AudioRecognition-master/Test_data/{}'.format(wav_path.split('/')[-1]))
    return 0

if __name__ == "__main__":
    files = os.listdir('/..1')
    #files.remove('.DS_Store')
    for f in files:
        print(f)
        feature0 = getFeatures('/../{}'.format(f))

    #print(feature0.shape)
    #plt.plot(feature0)
    #plt.show()
