import cv2
import mediapipe as mp
import time
from enum import IntEnum
import numpy as np
import math
import keyboard





""" TODO:
            get minimum viable product working
                music
                present line
                moving circles
                timed motion
                timed checking
                late/early/good/great UI text

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




class Dance:
    def __init__(self, name:str, song_path:str, poses:list):
        self.name = name
        self.song_path = song_path
        self.poses = poses
        

        self.pose_index = 1
        self.score = 0
        
        # used to calculate framerate
        self.cTime = 0
        self.pTime = 0
        
    def play_song(self):
        pass
    def check_poses(self):
        # if the player is doing the curret pose correctly
        if self.poses[self.pose_index].check():
            # lines and circles turn green
            mpDraw.draw_landmarks(flip, points, mpPose.POSE_CONNECTIONS, green_line_spec, green_line_spec)
            self.score += 1
            if self.pose_index < len(self.poses) -1:
                self.pose_index += 1
            else:
                self.pose_index = 0

        # if the player is not doing the current pose correctly     
        else:
            # lines and circles are black
            mpDraw.draw_landmarks(flip, points, mpPose.POSE_CONNECTIONS, black_circle_spec, black_line_spec)
            
    def handle_game_window(self):
        current_pose = self.poses[self.pose_index]
        if self.pose_index == len(self.poses) - 1:
            next_pose = self.poses[0]
        else:
            next_pose = self.poses[self.pose_index+1]

        # this is where framerate is calculated
        self.cTime = time.time()
        fps = 1/(self.cTime-self.pTime)
        self.pTime = self.cTime

        # put the framerate on the screen
        cv2.putText(flip,str(fps),(10,70),cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)
        cv2.putText(flip,str(self.score),(10,170),cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)
        

        # make a game window as a white screen the same size as flip
        game_window = np.zeros((len(flip),len(flip[0]),3), np.uint8)
        game_window.fill(255)


        blank_window = np.zeros((len(flip),len(flip[0]),3), np.uint8)
        blank_window.fill(255)
    
        cv2.putText(game_window,"Coming next!",(0,35),cv2.FONT_HERSHEY_PLAIN,3, (0,225,255),3)
        game_window = paste(next_pose.img_path ,45,0, game_window, scaling = .4)
        cv2.putText(game_window,"Do This!",(0,270),cv2.FONT_HERSHEY_PLAIN,3, (0,225,255),3)
        game_window = paste(current_pose.img_path ,285,0, game_window, scaling = .4)
        # Combining the two different image frames in one window
        combined_window = np.hstack([flip,game_window])
        
        cv2.imshow(window_name, combined_window)

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


    # checks if the player is making the pose
    def check(self):
        # check each subPose
        for subPose in self.subPoses:
            # if any subPose is not correct, return false
            if not subPose.check():
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

    def check(self):
        # if the points array exists
        if points:

            if self.angle == None and self.distance != None:
                if self.relative_position != "":
                    return self.check_relative_position()         
                else:
                    return self.check_distance()
            if self.angle != None:
                return self.check_angle()
        return False

    def check_distance(self):
        # get the screen positions of p1 and p2
        p1 = np.array(get_BodyPoint_pos(self.p1))
        p2 = np.array(get_BodyPoint_pos(self.p2))

        # handle unsupported operand error if one of the points is missing
        if (None in p1) or (None in p2):
            return False
        
        euclidian_dist = np.sqrt(np.sum((p1 - p2)**2))
        # returns false if any subGesture is outside the threshold
        if euclidian_dist > self.distance:
            return False
        # otherwise returns true if within the threshold
        return True

    def check_relative_position(self):
        p1 = np.array(get_BodyPoint_pos(self.p1))
        p2 = np.array(get_BodyPoint_pos(self.p2))
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
    def check_angle(self):
        # get the screen positions of p1 and p2
        x1 = np.array(get_BodyPoint_pos(self.p1))
        x2 = np.array(get_BodyPoint_pos(self.p2))
        x3 = np.array(get_BodyPoint_pos(self.p3))
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

            #print(player_angle)


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


def get_BodyPoint_pos(bodyPoint):
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



def paste(filename:str, x:int, y:int, game_window, scaling = 1):
    sprite = cv2.imread(filename)


    width = int(sprite.shape[1] * scaling)
    height = int(sprite.shape[0] * scaling)
    dim = (width, height)
      
    # resize image
    sprite = cv2.resize(sprite, dim, interpolation = cv2.INTER_AREA)

    output_image = game_window.copy()
    output_image[x:x+len(sprite),y:y+len(sprite[0]),:] = sprite
    return output_image



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


YMCA = Dance("YMCA", "",[Y,M,C,M])


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
    # left wrist below left shoulder
    SubPose(BodyPoint.LEFT_WRIST, BodyPoint.NOSE,  distance = 0, relative_position = "above"),
    # left wrist left of left shoulder
    SubPose(BodyPoint.LEFT_WRIST, BodyPoint.RIGHT_SHOULDER,  distance = 0, relative_position = "left_of"),
    # left wrist below left shoulder
    SubPose(BodyPoint.RIGHT_WRIST, BodyPoint.RIGHT_SHOULDER,  distance = 0, relative_position = "below"),
    #
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
                        [disco_pointing_up_right,
                         disco_pointing_up_left,
                         disco_right_arm_extended])




### main game loop ###

playing = True

current_dance = you_should_be_dancing
        
while playing:
    success, img = cap.read()
    #imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flip = cv2.flip(img,1)

    results = pose.process(flip)
    points = results.pose_landmarks
    
    if points:
        current_dance.check_poses()

    current_dance.handle_game_window()
    
    if keyboard.is_pressed('space'):
        playing = False
        cv2.destroyAllWindows()
        exit()

    cv2.waitKey(1)
