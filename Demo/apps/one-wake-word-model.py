import os
import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import python_speech_features
import RPi.GPIO as GPIO
import time

from tflite_runtime.interpreter import Interpreter

# Parameters
debug_time = 1
debug_acc = 1
led_pin = 40
word_threshold = 0.4
rec_duration = 0.5
window_stride = 0.5
sample_rate = 48000
resample_rate = 8000
num_channels = 1
num_mfcc = 16
model_path = os.path.abspath('path/to/model.tflite')

# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)

# GPIO 
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)  # Start with LED off

# Variable to track LED state
led_state = False  # False = Off, True = On

# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs
    
    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))
    return resampled_signal, new_fs

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):
    global led_state  # Use the global variable to track LED state

    # Start timing for testing
    start = timeit.default_timer()
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)
    
    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    
    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # Compute features
    mfccs = python_speech_features.base.mfcc(
        window,
        samplerate=new_fs,
        winlen=0.256,
        winstep=0.050,
        numcep=16,
        nfilt=26,
        nfft=2048,
        preemph=0.0,
        ceplifter=0,
        appendEnergy=False,
        winfunc=np.hanning
    )

    # Transpose to match expected format
    mfccs = mfccs[:16, :16].transpose()

    # Create a 128x128 array of zeros and insert the MFCCs
    padded_mfccs = np.zeros((128, 128), dtype=np.float32)
    padded_mfccs[:mfccs.shape[0], :mfccs.shape[1]] = mfccs

    # Use padded_mfccs as input to the model
    in_tensor = padded_mfccs.reshape(1, 128, 128, 1).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0][0]
    
    # When the word "on" is detected (threshold > word_threshold)
    if val > word_threshold:
        print('go')
        # Toggle LED state
        led_state = not led_state  # Flip the state
        GPIO.output(led_pin, GPIO.HIGH if led_state else GPIO.LOW)
        print(f"LED {'ON' if led_state else 'OFF'}")  # Debugging output to check LED state
        
    if debug_acc:
        print(val)
    
    if debug_time:
        print(timeit.default_timer() - start)

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass
