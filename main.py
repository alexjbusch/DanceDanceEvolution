import cv2
import time
from enum import IntEnum
import numpy as np
import math
import keyboard
import copy
import mediapipe as mp

# local scripts
from dance_dance_evolution import *
from utils import *

""" TODO:


"""

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
window_name = "dance dance Evolution"
cv2.namedWindow(window_name,cv2.WND_PROP_FULLSCREEN)
# set it to fullscreen
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

# points will become the list of points the ai detects on the player's body
points = None


### pose instantiation and definition ###
Y = Pose("",
    [
    # left arm straight
    SubPose(BodyPoint.LEFT_WRIST, BodyPoint.LEFT_ELBOW,  BodyPoint.LEFT_SHOULDER, angle = 360, lower_angle_threshold = 30, upper_angle_threshold = 30),
    # right arm straight
    SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.RIGHT_ELBOW,  BodyPoint.RIGHT_SHOULDER, angle = 360, lower_angle_threshold = 30, upper_angle_threshold = 30),

    # left upper arm angle between 290-250 degrees
    SubPose(BodyPoint.LEFT_HIP, BodyPoint.LEFT_SHOULDER,  BodyPoint.LEFT_ELBOW, angle = 300, lower_angle_threshold = 10, upper_angle_threshold = 50),
    # right upper arm angle between 10-70 degrees
    SubPose(BodyPoint.RIGHT_HIP, BodyPoint.RIGHT_SHOULDER,  BodyPoint.RIGHT_ELBOW, angle = 60, lower_angle_threshold = 50, upper_angle_threshold = 10),

    ])

M = Pose("",
    [
     #right elbow raised
     SubPose(BodyPoint.RIGHT_ELBOW, BodyPoint.RIGHT_SHOULDER,  BodyPoint.RIGHT_HIP, angle = 360, lower_angle_threshold = 75, upper_angle_threshold = 60),
     #left elbow raised
     SubPose(BodyPoint.LEFT_ELBOW, BodyPoint.LEFT_SHOULDER,  BodyPoint.LEFT_HIP, angle = 360, lower_angle_threshold = 75, upper_angle_threshold = 60),

     # right forearm angle between 75 and 180
     SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.RIGHT_ELBOW,  BodyPoint.RIGHT_SHOULDER, angle = 150, lower_angle_threshold = 75, upper_angle_threshold = 30),
     # left forearm angle between 190 and 285
     SubPose(BodyPoint.LEFT_WRIST, BodyPoint.LEFT_ELBOW,  BodyPoint.LEFT_SHOULDER, angle = 210, lower_angle_threshold = 20, upper_angle_threshold = 75),
    
    ])


C = Pose("",
    [
    # left wrist below right wrist
    SubPose(BodyPoint.LEFT_WRIST, BodyPoint.RIGHT_WRIST,  distance = 0, relative_position = "below"),
    # right wrist left of right shoulder
    SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.NOSE,  distance = 0, relative_position = "left_of"),
    # left wrist below nose
    SubPose(BodyPoint.LEFT_WRIST, BodyPoint.NOSE,  distance = 0, relative_position = "below"),
    ])


YMCA = Dance("YMCA", "",[(Y,5),(M,10),(C,15),(M,20)])


disco_pointing_down_left = Pose("",
    [
    # right wrist below left shoulder
    SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.LEFT_SHOULDER,  distance = 0, relative_position = "below"),
    #
    SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.NOSE,  distance = 0, relative_position = "left_of"),
    ])

disco_pointing_up_right = Pose("travolta_up_right.png",
    [
    # right wrist below left shoulder
    SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.NOSE,  distance = 0, relative_position = "above"),
    # right wrist right of right shoulder
    SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.RIGHT_SHOULDER,  distance = 0, relative_position = "right_of"),
    # left wrist below left shoulder
    SubPose(BodyPoint.LEFT_WRIST, BodyPoint.LEFT_SHOULDER,  distance = 0, relative_position = "below"),
    #
    SubPose(BodyPoint.LEFT_WRIST, BodyPoint.LEFT_ELBOW, BodyPoint.LEFT_SHOULDER, angle = 45, lower_angle_threshold = 20, upper_angle_threshold = 75),
    ])

disco_pointing_up_left = Pose("travolta_up_left.png",
    [
    # left wrist above nose
    SubPose(BodyPoint.LEFT_WRIST, BodyPoint.NOSE,  distance = 0, relative_position = "above"),
    # left wrist left of right shoulder
    SubPose(BodyPoint.LEFT_WRIST, BodyPoint.RIGHT_SHOULDER,  distance = 0, relative_position = "left_of"),
    # right wrist below right shoulder
    SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.RIGHT_SHOULDER,  distance = 0, relative_position = "below"),
    # right wrist at 270 degree angle to right shoulder
    SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.RIGHT_ELBOW, BodyPoint.RIGHT_SHOULDER, angle = 270, lower_angle_threshold = 20, upper_angle_threshold = 75),
    ])

disco_right_arm_extended = Pose("travolta_arm_right.png",
    [
    # right wrist below nose
    SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.NOSE,  distance = 0, relative_position = "below"),
    SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.RIGHT_HIP,  distance = 0, relative_position = "above"),
    SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.RIGHT_SHOULDER,  distance = 0, relative_position = "right_of"),
    #
    SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.RIGHT_ELBOW, BodyPoint.RIGHT_SHOULDER, angle = 0, lower_angle_threshold = 20, upper_angle_threshold = 20),
    SubPose(BodyPoint.RIGHT_HIP, BodyPoint.RIGHT_SHOULDER, BodyPoint.RIGHT_ELBOW, angle = 90, lower_angle_threshold = 30, upper_angle_threshold = 20),

    ])

you_should_be_dancing = Dance("You Should Be Dancing", "",
                        [(disco_pointing_up_right,5),
                         (disco_pointing_up_left,6),
                         (disco_right_arm_extended,7),
                         (copy.deepcopy(disco_pointing_up_right),9),
                        (copy.deepcopy(disco_pointing_up_left),10),
                        (copy.deepcopy(disco_right_arm_extended),11)])






### main game loop ###
playing = True

current_dance = you_should_be_dancing
print("Loop started")

while playing:
    success, img = cap.read()
    #imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # the flipped version of your webcam image (so that your reflection on screen isn't mirrored)
    flip = cv2.flip(img,1)

    results = pose.process(flip)
    points = results.pose_landmarks
    
    if points:
        current_dance.check_poses(flip,points)
        
    game_window = current_dance.handle_game_window(flip)
    cv2.imshow(window_name, game_window) 
    
    if keyboard.is_pressed('space'):
        playing = False
        cv2.destroyAllWindows()
        exit()

    cv2.waitKey(1)
