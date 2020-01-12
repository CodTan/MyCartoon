# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:08:11 2020

@author: DL
"""

# Import os to set the environment variable CUDA_VISIBLE_DEVICES
import os
import tensorflow as tf
import GPUtil

# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Get the first available GPU
DEVICE_ID_LIST = GPUtil.getFirstAvailable()
DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list

# Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

# Since all other GPUs are masked out, the first available GPU will now be identified as GPU:0
device = '/gpu:0'
print('Device ID (unmasked): ' + str(DEVICE_ID))
print('Device ID (masked): ' + str(0))

# Run a minimum working example on the selected GPU
# Start a session
with tf.Session() as sess:
    # Select the device
    with tf.device(device):
        # Declare two numbers and add them together in TensorFlow
        a = tf.constant(12)
        b = tf.constant(30)
        result = sess.run(a+b)
        print('a+b=' + str(result))