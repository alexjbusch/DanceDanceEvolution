import cv2
import time
from enum import IntEnum
import numpy as np
import math
import keyboard
import copy
import mediapipe as mp

import json

from dance_dance_evolution import *


### global declarations ###
cap = cv2.VideoCapture(0)

print(cap)
# note: mp.solutions.pose.Pose is a different class than my class called Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()
print(pose)
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



image_number = 1


def calculate_pose_attributes(im):
    results = pose.process(im)
    points = results.pose_landmarks
    
    relative_positions = {}
    angles = {
        "left_forearm_angle":0,
        "left_arm_angle":0,
        "right_forearm_angle":0,
        "right_arm_angle":0
             }
    points_of_comparison = [
                            BodyPoint.NOSE,
                            BodyPoint.RIGHT_WRIST,
                            BodyPoint.RIGHT_ELBOW,
                            BodyPoint.RIGHT_SHOULDER,
                            BodyPoint.RIGHT_HIP,
                            BodyPoint.LEFT_WRIST,
                            BodyPoint.LEFT_ELBOW,
                            BodyPoint.LEFT_SHOULDER,
                            BodyPoint.LEFT_HIP
                            ]







    for body_point_1 in points_of_comparison:
        for body_point_2 in points_of_comparison:
            p1 = np.array(get_BodyPoint_pos(points,im,body_point_1))
            p2 = np.array(get_BodyPoint_pos(points,im,body_point_2))
            try:
                if p1[1] < p2[1]:
                    relative_positions[body_point_1] = {body_point_2:["above"]}
                if p1[1] > p2[1]:
                    relative_positions[body_point_1] = {body_point_2:["below"]}
                    
                if p1[0] < p2[0]:
                    relative_positions[body_point_1][body_point_2].append("left_of")
                if p1[0] > p2[0]:
                    relative_positions[body_point_1][body_point_2].append("right_of")
            except IndexError:
                pass




    
    x1 = np.array(get_BodyPoint_pos(points,im,BodyPoint.RIGHT_WRIST))
    x2 = np.array(get_BodyPoint_pos(points,im,BodyPoint.RIGHT_ELBOW))
    x3 = np.array(get_BodyPoint_pos(points,im,BodyPoint.RIGHT_SHOULDER))
    right_forearm_angle = get_angle((x1,x2),(x2,x3))

    angles["right_forearm_angle"] = right_forearm_angle

    x1 = np.array(get_BodyPoint_pos(points,im,BodyPoint.RIGHT_ELBOW))
    x2 = np.array(get_BodyPoint_pos(points,im,BodyPoint.RIGHT_SHOULDER))
    x3 = np.array(get_BodyPoint_pos(points,im,BodyPoint.RIGHT_HIP))
    right_arm_angle = get_angle((x1,x2),(x2,x3))

    angles["right_arm_angle"] = right_arm_angle

    x1 = np.array(get_BodyPoint_pos(points,im,BodyPoint.LEFT_WRIST))
    x2 = np.array(get_BodyPoint_pos(points,im,BodyPoint.LEFT_ELBOW))
    x3 = np.array(get_BodyPoint_pos(points,im,BodyPoint.LEFT_SHOULDER))
    left_forearm_angle = get_angle((x1,x2),(x2,x3))

    angles["left_forearm_angle"] = left_forearm_angle


    x1 = np.array(get_BodyPoint_pos(points,im,BodyPoint.LEFT_ELBOW))
    x2 = np.array(get_BodyPoint_pos(points,im,BodyPoint.LEFT_SHOULDER))
    x3 = np.array(get_BodyPoint_pos(points,im,BodyPoint.LEFT_HIP))
    left_arm_angle = get_angle((x1,x2),(x2,x3))

    angles["left_arm_angle"] = left_arm_angle


    return (relative_positions, angles)





### main game loop ###




"""
im = cv2.imread("image_1.png")
results = pose.process(im)
points = results.pose_landmarks
#mpDraw.draw_landmarks(im, points, mpPose.POSE_CONNECTIONS, green_line_spec, green_line_spec)
#cv2.imshow(window_name,im)

attributes = calculate_pose_attributes(points,im)


points_of_comparison, angles = attributes
"""





def save_new_pose(attributes):
    subposes = []

    f = open(f'custom_pose_{image_number}.json', 'w')
    json.dump(attributes,f, indent=4)
    f.close()



playing = True
counting_down = False
start_time = time.time()
current_time = time.time()

while playing:
    success, img = cap.read()
    flip = cv2.flip(img,1)
    half_screen_width = int(len(flip)/2)
    half_screen_height = int(len(flip[0])/2)
    

    
    
    if keyboard.is_pressed('space'):
        playing = False
        cv2.destroyAllWindows()
        exit()

    if counting_down == False:
        if keyboard.is_pressed('enter'):
            print("enter")
            start_time = time.time()
            counting_down = True
        


    if counting_down: 
        current_time = time.time()
        if 3  - (current_time - start_time) < 0:
            start_time = time.time()
            
            try:
                attributes = calculate_pose_attributes(flip)
                save_new_pose(attributes)
                cv2.imwrite(f"pose_images/custom_pose_{image_number}.png", flip)
                image_number += 1
                flip[flip<255] = 255
                cv2.imshow(window_name, flip)
                time.sleep(.1)
                
            except (IndexError, AttributeError,KeyError) as error:
                cv2.putText(flip, "Pose Unclear!", (half_screen_width-100,half_screen_height),cv2.FONT_HERSHEY_PLAIN,4, (0,0,255),3)
                cv2.imshow(window_name, flip)
                cv2.waitKey(1)
                time.sleep(.2)

            counting_down = False
        else:

            text = str(int(4 - (current_time-start_time)))
            
            cv2.putText(flip, text, (half_screen_width,half_screen_height),cv2.FONT_HERSHEY_PLAIN,10, (255,255,255),3)

            
    cv2.imshow(window_name, flip)
    cv2.waitKey(1)
