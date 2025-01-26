import os
import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import python_speech_features
import RPi.GPIO as GPIO

from tflite_runtime.interpreter import Interpreter

# Parameters
debug_mode = False  # Set to False to suppress detailed debug prints
led_pin = 40
word_threshold = 0.4
rec_duration = 0.5
sample_rate = 48000
resample_rate = 8000
num_channels = 1
num_mfcc = 16
model_path = os.path.abspath('path/to/model.tflite')  # Path to your "on/go" model

# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)

# GPIO 
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)  # Start with LED off

# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs
    
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))
    return resampled_signal, new_fs

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):
    # Start timing for processing time measurement
    start = timeit.default_timer()
    
    # Notify if errors
    if status and debug_mode:
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
    
    mfccs = mfccs[:16, :16]
    mfccs = mfccs.transpose()

    # Prepare input tensor for the model
    in_tensor = np.float32(mfccs.reshape(1, 16, 16, 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get probabilities for "on" and "go"
    val_on = output_data[0][0]  # Probability of "on"
    val_go = output_data[0][1]  # Probability of "go"

    # LED Control
    if val_on > word_threshold:
        GPIO.output(led_pin, GPIO.HIGH)  # Turn LED on
        print("Recognized: ON -> LED ON")
    elif val_go > word_threshold:
        GPIO.output(led_pin, GPIO.LOW)  # Turn LED off
        print("Recognized: GO -> LED OFF")
    
    # Debug information (optional)
    if debug_mode:
        print(f"ON: {val_on}, GO: {val_go}")
        print("Processing time:", timeit.default_timer() - start)

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    print("Listening for 'on' or 'go'...")
    while True:
        pass
