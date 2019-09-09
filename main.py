#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import queue
import sys
import threading

from PySide2.QtCore import QObject, Signal, Slot
from PySide2.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PySide2.QtMultimedia import QAudioFormat, QAudioDeviceInfo, QAudioInput

from deepspeech import Model
import numpy as np


N_FEATURES = 26
N_CONTEXT = 9
BEAM_WIDTH = 500
LM_ALPHA = 0.75
LM_BETA = 1.85


class InferenceThread(QObject):
    finished = Signal(str)

    def __init__(self, model, alphabet, lmbin, trie):
        super(InferenceThread, self).__init__()
        self._in_queue = queue.Queue()
        self._should_quit = False
        self._worker = threading.Thread(target=self.run,
                                        args=(model, alphabet, lmbin, trie))

    def send_cmd(self, cmd):
        ''' Insert command in queue to be processed by the thread '''
        self._in_queue.put(cmd)

    def set_quit(self):
        ''' Signal to the thread that it should stop running '''
        self._should_quit = True

    def start(self):
        self._worker.start()

    def run(self, model, alphabet, lmbin, trie):
        model = Model(model, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)
        model.enableDecoderWithLM(alphabet, lmbin, trie, LM_ALPHA, LM_BETA)
        stream = None

        while True:
            # Try to get the next command from our queue, use a timeout to check
            # periodically for a quit signal so the application doesn't hang on
            # exit.
            try:
                cmd, *data = self._in_queue.get(timeout=0.3)
            except queue.Empty:
                if self._should_quit:
                    break
                # If we haven't received a quit signal just continue trying to
                # get a command from the queue indefinitely
                continue

            if cmd == 'start':
                # 'start' means create a new stream
                stream = model.setupStream()
            elif cmd == 'data':
                # 'data' means we received more audio data from the recorder
                if stream:
                    model.feedAudioContent(stream, np.frombuffer(data[0].data(), np.int16))
            elif cmd == 'finish':
                # 'finish' means the caller wants the result of the current stream
                transcript = model.finishStream(stream)
                self.finished.emit(transcript)
                stream = None


class Dialog(QWidget):
    def __init__(self, inference_thread):
        super(Dialog, self).__init__()
        self._btn = QPushButton('Record', self)
        self._label = QLabel('', self)

        layout = QVBoxLayout()
        layout.addWidget(self._btn)
        layout.addWidget(self._label)
        self.setLayout(layout)

        self._btn.clicked.connect(self._btn_clicked)
        self._is_recording = False

        self._inference_thread = inference_thread
        self._inference_thread.finished.connect(self._on_transcription_finished)

        audio_format = QAudioFormat()
        audio_format.setCodec('audio/pcm')
        audio_format.setChannelCount(1)
        audio_format.setSampleSize(16)
        audio_format.setSampleRate(16000)
        audio_format.setByteOrder(QAudioFormat.LittleEndian)
        audio_format.setSampleType(QAudioFormat.SignedInt)

        input_device_info = QAudioDeviceInfo.defaultInputDevice()
        if not input_device_info.isFormatSupported(audio_format):
            print('Can\'t record audio in 16kHz 16-bit signed PCM format.')
            exit(1)

        self._audio_input = QAudioInput(audio_format)

    @Slot()
    def _btn_clicked(self):
        if self._is_recording:
            # Was recording -> Button clicked -> Stop recording
            self._is_recording = False
            self._btn.setText('Record')
            self._audio_input.stop()
            self._inference_thread.send_cmd(('finish',))
        else:
            # Was not recording -> Button clicked -> Start recording
            self._is_recording = True
            self._label.setText('...')
            self._btn.setText('Stop')
            self._inference_thread.send_cmd(('start',))
            # QAudioInput retains the QIODevice returned here internally so
            # there's no need to keep a reference to it
            io_device = self._audio_input.start()
            io_device.readyRead.connect(self._read_from_io_device)

    @Slot()
    def _read_from_io_device(self):
        ''' Forward available audio data to the inference thread. '''
        # self.sender() is the IO device returned by QAudioInput.start()
        self._inference_thread.send_cmd(('data', self.sender().readAll()))

    @Slot(str)
    def _on_transcription_finished(self, result):
        print('Transcription:', result)
        self._label.setText(result)


def main():
    parser = argparse.ArgumentParser(description='Streaming speech-to-text using deepspeech and PySide2')
    parser.add_argument('--model', required=True, help='Path to the model (protocol buffer binary file, .pb or .pbmm or .tflite)')
    parser.add_argument('--alphabet', required=True, help='Path to the configuration file specifying the alphabet used by the network (alphabet.txt)')
    parser.add_argument('--lm', nargs='?', help='Path to the language model binary file (lm.binary)')
    parser.add_argument('--trie', nargs='?', help='Path to the language model trie file created with native_client/generate_trie (trie)')
    args, unknown_args = parser.parse_known_args()

    # Create inference thread
    inference_thread = InferenceThread(args.model, args.alphabet, args.lm, args.trie)

    app = QApplication(unknown_args)
    dialog = Dialog(inference_thread)
    dialog.show()

    # Start inference thread
    inference_thread.start()

    # Run Qt main loop
    ret = app.exec_()

    # Signal to inference thread that the application is quitting
    inference_thread.set_quit()

    sys.exit(ret)


if __name__ == '__main__':
    main()
