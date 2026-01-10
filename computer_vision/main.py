"""
This script implements openCV libary in python to detect moving objects and draw graphical symbols on them in a video stream.

All required packages are listed in the requirements.txt file. Be mindful to install them in your python environment before running the script. Packages versions matter, so use those specified in the file. Newer versions of Mediapipe use Tasks API instead of Solutions.

Authors:
- Fabian Fetter
- Konrad Fijałkowski


Usage:
Run the script from the command line in the same directory as the main.py file. 

"""
import cv2
import mediapipe as mp
import math

class MotionDetector:
    """
    A class to detect motion based on body pose estimation using MediaPipe.
    """
    def __init__(self):
        """
        Initializes the MotionDetector with default settings.
        """
        self.source_stream = 'motion.mov'
        self.previous_nose_position = None
        self.current_nose_position = None
        self.moving_threshold_pixels = 2
        self.draw_pose_landmarks = False
        self.frame = None
        self.processed_pose = None

    def quit_on_keypress(self, key):
        """
        Wait 20 miliseconds for a keypress
        """
        if cv2.waitKey(20) == ord(key):
            exit()

    def draw_landmarks(self):
        """
        Draw found landsmarks on a body using MediaPipe.
        """
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils 

        mp_drawing.draw_landmarks(
                self.frame, 
                self.processed_pose.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
                )
        
    def print_text_on_frame(self, text, font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=3):
        """
        Put specified text onto the video stream.
        """
        cv2.putText(self.frame, text, (50, 100), font, 2, color, thickness)
        

    def draw_crosshair_on_face(self, color=(0, 0, 255), thickness=3):
        """
        Create crosshair on target's face.
        """
        h, w, _ = self.frame.shape
        ear = self.processed_pose.pose_landmarks.landmark[7]
        ear_x, ear_y = int(ear.x * w), int(ear.y * h)

        nx, ny = self.current_nose_position

        face_radius = int(math.dist((nx, ny), (ear_x, ear_y)) * 2)

        cv2.circle(self.frame, (nx, ny), face_radius, color, thickness)

        tick_len = 10
        cv2.line(self.frame, (nx, ny - face_radius - tick_len), (nx, ny - face_radius + tick_len), color, thickness) #top
        cv2.line(self.frame, (nx, ny + face_radius + tick_len), (nx, ny + face_radius - tick_len), color, thickness) #bottom
        cv2.line(self.frame, (nx + face_radius - tick_len, ny), (nx + face_radius + tick_len, ny), color, thickness) #right
        cv2.line(self.frame, (nx - face_radius - tick_len, ny ), (nx - face_radius + tick_len, ny), color, thickness) #left

    def print_text_on_move(self):
        """
        Print text whenever target body moves in the video stream
        """
        if self.previous_nose_position is not None:
            distance_between_frames = math.dist(self.previous_nose_position, self.current_nose_position)

            if distance_between_frames >= self.moving_threshold_pixels:
                self.print_text_on_frame('SHOOT', color=(0, 255, 0))
                self.draw_crosshair_on_face()
            else:
                self.print_text_on_frame('HOLD', color=(0, 0, 255))

    def detect_motion(self):
        """
        Detect in a while loop if body on a video stream moved. Add pose landmarks and show output to an additional window.
        """
        pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(self.source_stream)

        while cap.isOpened():
            frame_read_success, self.frame = cap.read()

            if not frame_read_success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.processed_pose = pose.process(rgb_frame)
            frame_height, frame_width, _ = self.frame.shape

            if self.processed_pose.pose_landmarks:

                nose_landmark = self.processed_pose.pose_landmarks.landmark[0]
                nose_x, nose_y = int(nose_landmark.x * frame_width), int(nose_landmark.y * frame_height)
                self.current_nose_position = (nose_x, nose_y)

                if self.draw_pose_landmarks: self.draw_landmarks()

                self.print_text_on_move()

                self.previous_nose_position = self.current_nose_position

            frame_percentage = 0.5
            cv2.namedWindow('CV motion capture', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('CV motion capture', int(frame_percentage*frame_width), int(frame_percentage*frame_height))
            cv2.imshow('CV motion capture', self.frame)

            self.quit_on_keypress('q')

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cv = MotionDetector()
    cv.detect_motion()
