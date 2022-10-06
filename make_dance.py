import cv2
import time
from enum import IntEnum
import numpy as np
import math
import keyboard
import copy
import mediapipe as mp
### global declarations ###
cap = cv2.VideoCapture(0)


# note: mp.solutions.pose.Pose is a different class than my class called Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

black_circle_spec = mpDraw.DrawingSpec(thickness=5, color=(0,0,0))
black_line_spec = mpDraw.DrawingSpec(thickness=1, color=(0,0,0))
green_line_spec = mpDraw.DrawingSpec(thickness=2, color=(0,255,0))
green_circle_spec = mpDraw.DrawingSpec(thickness=8, color=(0,255,0))

# create the window
window_name = "dance creator"
cv2.namedWindow(window_name,cv2.WND_PROP_FULLSCREEN)
# set it to fullscreen
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

# points will become the list of points the ai detects on the player's body
points = None


### main game loop ###
playing = True
start_time = None

image_number = 1

im = cv2.imread("image_1.png")
results = pose.process(im)
points = results.pose_landmarks
mpDraw.draw_landmarks(im, points, mpPose.POSE_CONNECTIONS, green_line_spec, green_line_spec)
cv2.imshow(window_name,im)
"""
while playing:
    success, img = cap.read()
    flip = cv2.flip(img,1)



    cv2.imshow(window_name, flip)
    
    if keyboard.is_pressed('space'):
        playing = False
        cv2.destroyAllWindows()
        exit()

    if keyboard.is_pressed('enter'):
        print("enter")
        start_time = time.time()

    if start_time:
        if time.time() - start_time > 2:
            start_time = None
            cv2.imwrite(f"image_{image_number}.png", flip)
            image_number += 1
    
    cv2.waitKey(1)
"""
