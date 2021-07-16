"""
2021.07.14 생성됨
파일위치 : /Users/seungyeonkoo/Documents/2021summerinternship
참고자료 : https://youdaeng-com.tistory.com/5
virtualenv : /Users/seungyeonkoo/Documents/2021summerinternship/venvhesit
- 침묵구간 추출
- 침묵구간 추출 검증
"""

# DATA_DIR_TRAIN = '/Users/seungyeonkoo/Documents/audio/hesitation/SpeakUP-git/dataset/train/'
# DATA_DIR_TEST = '/Users/seungyeonkoo/Documents/audio/hesitation/SpeakUP-git/dataset/test/'
DATA_DIR = '/Users/seungyeonkoo/Documents/audio/data_set_stt/'
OUTPUT_DIR = '/Users/seungyeonkoo/Documents/2021summerinternship/temp/'
file_name = 'case0_2_5_nohash_.wav'

# import librosa
# import sklearn
from pydub.silence import detect_nonsilent
from pydub import AudioSegment
import matplotlib.pyplot as plt
import os



# audio, sr = librosa.load(audio_file, sr=16000)
# print('sr:', sr, ', audio shape:', audio.shape)
# print('length:', audio.shape[0]/float(sr), 'secs')

# mfcc = librosa.feature.mfcc(audio, sr=16000, n_mfcc=100, n_fft=400, hop_length=160)
# mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

file_list = sorted(os.listdir(DATA_DIR))
file_list_wav = [file for file in file_list if file.endswith(".wav")]
for file_name in file_list_wav[:5] :
    print('test file list : ', file_list_wav[:5])
    audio_file = DATA_DIR + file_name
    audio = AudioSegment.from_wav(audio_file)
    min_silence_length = 70
    intervals = detect_nonsilent(audio, min_silence_len=min_silence_length, silence_thresh=-32.64)
    print('file directory : ', DATA_DIR)
    print('file name : ', file_name)
    print('intervals :')
    print(intervals)

    # 침묵구간이 잘 추출 됐는지 검증
    # num_files = len(intervals) + 1
    output = audio
    output.export(OUTPUT_DIR + file_name, format='wav')
    lastend = 0
    for interval in enumerate(intervals) :
        start = []
        end = []

        start.append(lastend)
        end.append(interval[1][0])

        start.append(interval[1][0])
        end.append(interval[1][1])

        if interval[0] == len(intervals) - 1 and interval[1][1] < len(audio) :
            start.append(interval[1][1])
            end.append(len(audio))
        print('\n<make wav file>')
        for i in range(len(start)) :
            output = audio[start[i] : end[i]]
            print('interval:', interval, ', i:', i, ', number(interval[0]*2+i):', interval[0]*2+i)
            print('start[i] : ', start[i], ', end[i] : ', end[i])
            output_filename = OUTPUT_DIR+file_name[:-11]+\
                              'temp'+str(interval[0]*2+i)+'.wav'
            output.export(output_filename, format='wav')

        lastend = interval[1][1]

# plot
# # raw wave
# print('audio :\n', audio)
# print('output :\n', output)
# plt.plot(output)
# plt.show()