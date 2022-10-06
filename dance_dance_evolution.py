import cv2

import time
from enum import IntEnum
import numpy as np
import math
import keyboard
import copy

import mediapipe as mp

class Text:
    def __init__(self, text:str, coordinates, lifetime=2, speed = 1.5, color=(0,0,0)):
        self.text = text
        self.x, self.y = coordinates
        self.lifetime = lifetime
        self.speed = speed
        self.color = color
        
        self.start_time = time.time()
        
    def move(self):
        self.y -= self.speed
        # make sure pixels are int values
        self.y = round(self.y)
        self.x = round(self.x)
        if time.time()-self.start_time > self.lifetime or self.y < 0:
            return False
        return True
