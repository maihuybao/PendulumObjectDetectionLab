# import torch
# # from ultralytics import YOLO

# # Check for CUDA device and set it
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(torch.cuda.get_device_name(0))

# import cv2
# # check gpu
# # print(cv2.cuda.getCudaEnabledDeviceCount())
# # version

from math import sqrt

def calculate_T(length):
    return round(2 * 3.14 * sqrt(length / 9.8), 2)
print(calculate_T(0.8))