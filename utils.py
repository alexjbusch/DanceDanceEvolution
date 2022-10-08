import cv2

import time
from enum import IntEnum
import numpy as np
import math
import keyboard
import copy

import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

black_circle_spec = mpDraw.DrawingSpec(thickness=5, color=(0,0,0))
black_line_spec = mpDraw.DrawingSpec(thickness=1, color=(0,0,0))
green_line_spec = mpDraw.DrawingSpec(thickness=2, color=(0,255,0))
green_circle_spec = mpDraw.DrawingSpec(thickness=8, color=(0,255,0))

### classes and enums ###

class BodyPoint(IntEnum):
    # NOTE: ALL OF THE POSITIONS ARE FLIPPED BECAUSE THE IMAGE WILL BE FLIPPED
    NOSE = 0
    RIGHT_EYE_INNER = 1
    RIGHT_EYE = 2
    RIGHT_EYE_OUTER = 3
    LEFT_EYE_INNER = 4
    LEFT_EYE = 5
    LEFT_EYE_OUTER = 6
    RIGHT_EAR = 7
    LEFT_EAR = 8
    MOUTH_RIGHT = 9
    MOUTH_LEFT = 10

    RIGHT_SHOULDER = 11
    LEFT_SHOULDER = 12

    RIGHT_ELBOW = 13
    LEFT_ELBOW = 14

    RIGHT_WRIST = 15
    LEFT_WRIST = 16

    RIGHT_PINKY = 17
    LEFT_PINKY = 18
    RIGHT_INDEX = 19
    LEFT_INDEX = 20
    RIGHT_THUMB = 21
    LEFT_THUMB = 22

    RIGHT_HIP = 23
    LEFT_HIP = 24

    RIGHT_KNEE = 25
    LEFT_KNEE = 26

    RIGHT_ANKLE = 27
    LEFT_ANKLE = 28

    RIGHT_HEEL = 29
    LEFT_HEEL = 30

    RIGHT_FOOT_INDEX = 31
    LEFT_FOOT_INDEX = 32




### helper functions ###

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def cross(a,b):
    return math.atan2(a[0]*b[1] - a[1]*b[0], a[0]*b[0] + a[1]*b[1])

def get_angle(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]

    angle = cross(vA, vB)
    ang_deg = math.degrees(angle)%360
    return ang_deg


def get_BodyPoint_pos(points,flip,bodyPoint):
    x1 = points.landmark[bodyPoint].x
    y1 = points.landmark[bodyPoint].y
    image_rows, image_cols, _ = flip.shape
    screen_coords = mpDraw._normalized_to_pixel_coordinates(x1, y1,
                                               image_cols, image_rows)
    return screen_coords

def draw_line_between_landmarks(bodyPoint1, bodyPoint2):
    screen_coords1 = get_BodyPoint_pos(bodyPoint1)
    screen_coords2 = get_BodyPoint_pos(bodyPoint2)
    # draw the line between the two landmarks
    cv2.line(flip, screen_coords1, screen_coords2,(255,100,100),2)



def paste(filename:str, x:int, y:int, game_window, scalingx = 1, scalingy = 1):
    if x > len(game_window) or y > len(game_window[0]):
        return game_window
    if x < 0 or y < 0:
        return game_window
    
    sprite = cv2.imread(filename)

    sprite = cv2.resize(sprite, dsize=(444, 444), interpolation=cv2.INTER_CUBIC)

    width = int(sprite.shape[1] * scalingx)
    height = int(sprite.shape[0] * scalingy)
    dim = (width, height)
      
    # resize image
    sprite = cv2.resize(sprite, dim, interpolation = cv2.INTER_AREA)

    output_image = game_window.copy()

    if x+len(sprite[0]) <= len(output_image[0]):
        output_image[y:y+len(sprite),x:x+len(sprite[0]),:] = sprite
    else:
        image_difference = abs(len(output_image[0]) - x+len(sprite[0]))
        output_image[x:len(output_image),y:y+len(output_image[0]),:] = sprite[0:,0:,:]
    return output_image


