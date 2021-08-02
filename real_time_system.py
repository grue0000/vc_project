"""
2021.07.28 생성됨
- 실시간으로 보이스피싱인지 아닌지 화면에 표시해준다.
- update_time_interval는 7000의 약수를 권장
- update_time_interval이 너무 작아지면 실제 시간보다 많이 느려지는 문제 발생
- update_time_interval을 1000으로 해도 시간이 지날 수록 실제 시간보다 느려지는 문제 발생
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import micrec
import input_data
import models
import os

def print_time(cnt) :
    print('time : %s sec (cnt=%s)' % (cnt*FLAGS.update_time_interval/1000, cnt))

from pydub import AudioSegment
def print_filelength(filename) :
    wav_file = open(filename, 'rb')
    sample = AudioSegment.from_wav(wav_file)
    print('%s : file length : %s' % (filename, len(sample)))
    wav_file.close()

def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()
    words_list = FLAGS.wanted_words.split(',')
    model_settings = models.prepare_model_settings(
        len(words_list), FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)

    audio_processor = input_data.AudioProcessor(model_settings)

    fingerprint_size = model_settings['fingerprint_size']

    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits = models.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        FLAGS.model_size_info,
        is_training=False)

    predicted_indices = tf.argmax(logits, 1)
    models.load_variables_from_checkpoint(sess, FLAGS.checkpoint)

    cnt = 0
    frames=[]
    try :
        for _ in range(int(7000/FLAGS.update_time_interval)) :
            _, file_name = micrec.record(record_seconds=FLAGS.update_time_interval/1000, frames=frames)
            cnt += 1
            print_time(cnt)
            print_filelength(file_name)
            print('*')

        while True :
            inference_fingerprints = audio_processor.get_single_data(0, file_name, model_settings, sess)
            predicted_result, before_predict = sess.run(
                [predicted_indices, logits],
                feed_dict={
                    fingerprint_input: inference_fingerprints,
                })

            if predicted_result[0] == 0 :
                print('predicted result : 일반음성통화')
            elif predicted_result[0] == 1 :
                print('predicted result : 보이스피싱!!!!')

            frames.pop()
            _, file_name = micrec.record(record_seconds=FLAGS.update_time_interval/1000, frames=frames)
            cnt += 1
            print_time(cnt)
            print_filelength(file_name)

    except KeyboardInterrupt :
        print("\n*** End Process ***")
        os.remove(file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--update_time_interval',
        type=int,
        default=1000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=7000,
        help='Expected duration in milliseconds of the wavs', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint', )
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='0,1',
        help='Words to use (others will be added to an unknown label)', )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="/Users/seungyeonkoo/Documents/2021summerinternship/model/lstm_9729.ckpt-160",
        help='Checkpoint to load the weights from.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='lstm',
        help='What model architecture to use')
    parser.add_argument(
        '--model_size_info',
        type=int,
        nargs="+",
        default=[2, 2000],
        help='Model dimensions - different for various models')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)