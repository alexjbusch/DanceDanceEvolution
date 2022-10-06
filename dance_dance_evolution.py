import cv2

import time
from enum import IntEnum
import numpy as np
import math
import keyboard
import copy

import mediapipe as mp

from utils import *

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







class Dance:
    def __init__(self, name:str, song_path:str, poses:list, speed=300):
        self.name = name
        self.song_path = song_path
        self.poses = poses

        self.original_pose_list = copy.deepcopy(poses)

        self.current_pose = self.poses[0][0]

        # a list of Text objects
        self.text_list = []

        # measured in pixels per second
        self.speed = speed
        self.start_time = time.time()
        
        self.score = 0
        
        # used to calculate framerate
        self.cTime = 0
        self.pTime = 0
        
    def play_song(self):
        pass

    def pose_was_failed(self):
        current_time = time.time() - self.start_time
        time_to_do_current_pose = self.poses[0][1]
        if time_to_do_current_pose - current_time <= 0:
            self.poses.pop(0)

            if not self.poses:
                self.poses = copy.deepcopy(self.original_pose_list)
                self.start_time = time.time()
            self.current_pose = self.poses[0][0]
            return True
        return False
    
    def check_poses(self,img,points):
        # if the player is doing the curret pose correctly
        if self.current_pose:

            if self.current_pose.check(img,points):
                 # lines and circles turn green
                mpDraw.draw_landmarks(img, points, mpPose.POSE_CONNECTIONS, green_line_spec, green_line_spec)
                self.score += 1
                self.poses.pop(0)

                
                good_text = Text("Good!",(int(self.current_pose.x),200),color=(0,255,0))
                self.text_list.append(good_text)
                
                # loops the dance infinitely
                if not self.poses:
                    self.poses = copy.deepcopy(self.original_pose_list)
                    self.start_time = time.time()
                    
                self.current_pose = self.poses[0][0]

            # if the player is not doing the current pose correctly     
            else:
                # lines and circles are black
                mpDraw.draw_landmarks(img, points, mpPose.POSE_CONNECTIONS, black_circle_spec, black_line_spec)
            
    def handle_game_window(self,input_image):

        
        # this is where framerate is calculated
        self.cTime = time.time()
        fps = 1/(self.cTime-self.pTime)
        self.pTime = self.cTime

        # put the framerate on the screen
        cv2.putText(input_image,str(fps),(10,70),cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)
        cv2.putText(input_image,str(self.score),(10,170),cv2.FONT_HERSHEY_PLAIN,3, (0,255,0),3)            

        # make a game window as a white screen the same size as the webcam image
        game_window = np.zeros((len(input_image),len(input_image[0]),3), np.uint8)
        game_window.fill(255)


        if self.pose_was_failed():
            bad_text = Text("Bad!",(20,200),color=(0,0,255))
            self.text_list.append(bad_text)

        for text in self.text_list:
            if text.move():
                cv2.putText(game_window, text.text, (text.x,text.y),cv2.FONT_HERSHEY_PLAIN,3, text.color,3)
            else:
                self.text_list.remove(text)


        
        current_time = time.time() - self.start_time
        for pose,time_to_do_pose in self.poses:
            pose.x = time_to_do_pose *self.speed - (current_time*self.speed)
            game_window = paste(pose.img_path ,int(pose.x) ,pose.y, game_window, scalingx = .3, scalingy = .7)
            

        # Combining the two different image frames in one window
        combined_window = np.vstack([game_window,input_image])
        
        return combined_window



""" Poses are made up of subPoses
    subPoses is a list of SubPose objects
    SubPoses can be:

    a distance between two bodyPoints,
    a distance between a bodypoint and an objective point,
    an angle between two lines,
    a distance above/below/left of/right of another point
    
"""
class Pose:
    def __init__(self, img_path:str, subPoses:list):
        # the path pointing to an image of the pose
        self.img_path = "pose_images\\"+img_path
        # a list of subPose objects
        self.subPoses = subPoses

        self.x = 700
        self.y = 0

        self.time_to_do_pose = 5

    # checks if the player is making the pose
    def check(self,img,points):
        # check each subPose
        for subPose in self.subPoses:
            # if any subPose is not correct, return false
            if not subPose.check(img,points):
                return False
        # if they are all correct, return true
        return True

class SubPose:
    def __init__(self, p1, p2, p3 = None, distance = None, angle = None, upper_angle_threshold = 20, lower_angle_threshold = 20, relative_position = ""):
        # the first BodyPoint
        self.p1 = p1
        # the second BodyPoint
        self.p2 = p2
        # the third BodyPoint, only used to pass two lines
        self.p3 = p3
        # the maximum allowed distance between p1 and p2
        self.distance = distance
        # the target angle of the subPose
        self.angle = angle
        # the allowed deviation from the target angle that counts as being close enough to the pose
        self.upper_angle_threshold = upper_angle_threshold
        self.lower_angle_threshold = lower_angle_threshold

        # whether to check if p1 is above/below/left of/right of p2
        self.relative_position = relative_position

    def check(self,img,points):
        # if the points array exists
        if points:

            if self.angle == None and self.distance != None:
                if self.relative_position != "":
                    return self.check_relative_position(img,points)         
                else:
                    return self.check_distance(img,points)
            if self.angle != None:
                return self.check_angle(img,points)
        return False

    def check_distance(self,img,points):
        # get the screen positions of p1 and p2
        p1 = np.array(get_BodyPoint_pos(points,img,self.p1))
        p2 = np.array(get_BodyPoint_pos(points,img,self.p2))

        # handle unsupported operand error if one of the points is missing
        if (None in p1) or (None in p2):
            return False
        
        euclidian_dist = np.sqrt(np.sum((p1 - p2)**2))
        # returns false if any subGesture is outside the threshold
        if euclidian_dist > self.distance:
            return False
        # otherwise returns true if within the threshold
        return True

    def check_relative_position(self,img,points):
        p1 = np.array(get_BodyPoint_pos(points,img,self.p1))
        p2 = np.array(get_BodyPoint_pos(points,img,self.p2))
        if self.relative_position == "above":
            try:
                if p1[1] < p2[1] - self.distance:
                    return True
            except IndexError:
                return False
        elif self.relative_position == "below":
            try:
                if p1[1] > p2[1] + self.distance:
                    return True
            except IndexError:
                return False

        elif self.relative_position == "left_of":
            try:
                if p1[0] < p2[0] - self.distance:
                    return True
            except IndexError:
                return False

        elif self.relative_position == "right_of":
            try:
                if p1[0] > p2[0] + self.distance:
                    return True
            except IndexError:
                return False
        return False
            

    # checks whether the angle of two lines is within a given threshold of self.angle
    def check_angle(self,img,points):
        # get the screen positions of p1 and p2
        x1 = np.array(get_BodyPoint_pos(points,img,self.p1))
        x2 = np.array(get_BodyPoint_pos(points,img,self.p2))
        x3 = np.array(get_BodyPoint_pos(points,img,self.p3))
        # handle unsupported operand error if one of the points is missing
        if (None in x1) or (None in x2) or (None in x3):
            return False
        # returns true if the actual angle is within the angle threshold of the SubPose's angle
        try:
            
            player_angle = get_angle((x1,x2),(x2,x3))
            upper_bound = self.angle + self.upper_angle_threshold
            lower_bound = self.angle - self.lower_angle_threshold



            if upper_bound > 360:
                upper_bound -= 360
            if lower_bound < 0:
                lower_bound += 360

            # if the upper and lower bounds are flipped so the lower bound is higher than the upper bound
            if upper_bound < lower_bound:
                # change the AND to an OR
                if player_angle > lower_bound or player_angle < upper_bound:
                    return True
            # otherwise if the upper bound is higher than the lower bound
            elif lower_bound < upper_bound:
                # change the OR to an AND
                if player_angle > lower_bound and player_angle < upper_bound:
                    return True
            

            # otherwise returns false
            return False
        except ValueError:
            return False
