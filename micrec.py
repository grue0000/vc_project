"""
2021.07.23 생성됨
참고 자료 : https://javaexpert.tistory.com/677
- 함수 record()를 정의
"""
import pyaudio
import wave

def record (frames=[], record_seconds=7, filename="temp.wav", ruprint=False,
            FORMAT=pyaudio.paInt16, CHANNELS=1, RATE=16000, CHUNK=1600, INDEX=0) :
    """
    - record() :  마이크에서 들어온 음성 신호 녹음하는 코드
    - 이 파일을 실행하기 전 아래와 같이 마이크 인덱스 번호를 추출
        /Users/seungyeonkoo/Documents/2021summerinternship/venvRT/bin/python /Users/seungyeonkoo/Documents/2021summerinternship/mic_device_test.py
        DEVICE: MacBook Pro 마이크  INDEX:  0  RATE:  44100
        DEVICE: MacBook Pro 스피커  INDEX:  1  RATE:  44100
    """

    # start
    audio = pyaudio.PyAudio()
    if ruprint : print("start recording!")
    stream=audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=INDEX, frames_per_buffer=CHUNK)
    if ruprint : print("recording...")

    data = stream.read(int(RATE*record_seconds))
    frames.append(data)
    if ruprint : print("finished recording")

    # stop
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save
    waveFile = wave.open(filename,'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    return frames, filename