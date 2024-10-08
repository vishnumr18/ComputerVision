# main.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.audio import SoundLoader

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from ultralytics import YOLO
import operator as op
import time

class DrowsinessDetectionApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.img = Image()
        self.status_label = Label(text='Status: Not Started', size_hint=(1, 0.1))
        self.start_button = Button(text='Start', on_press=self.start_detection, size_hint=(1, 0.1))
        
        self.layout.add_widget(self.img)
        self.layout.add_widget(self.status_label)
        self.layout.add_widget(self.start_button)
        
        self.setup_drowsiness_detection()
        
        return self.layout

    def setup_drowsiness_detection(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, 
                                                    min_detection_confidence=0.5, 
                                                    min_tracking_confidence=0.5)
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        self.class_name = ["cell phone"]
        self.class_nameto_id = {value:key for key, value in self.model.names.items()}
        self.class_id = [self.class_nameto_id[name] for name in self.class_name]
        
        # Other initializations
        self.eyeClose = 0
        self.ear_threshold = 0.2
        self.max_frame_ec = 20
        self.max_frame_direction = 30
        self.dcounter = 0
        self.operators = {">": op.gt, "<": op.lt}
        self.direction = ">"
        
        # Head pose estimation parameters
        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        # Load beep sound
        self.beep_sound = SoundLoader.load('beep-01a.wav')
        self.last_beep_time = 0

    def start_detection(self, instance):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/30.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Process the frame
            frame = cv2.flip(frame, 1)
            frame = self.process_frame(frame)
            
            # Convert it to texture
            buf = cv2.flip(frame, 0).tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            
            # Display image from the texture
            self.img.texture = image_texture

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        half = w // 2
        cv2.line(frame, (half,0),(half, h), (255,255,0), 2)
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lcx = face_landmarks.landmark[227].x * w

                nose_lm = face_landmarks.landmark[1]
                print(nose_lm)
                nx = int(nose_lm.x*w)
                ny = int(nose_lm.y*h)
                cv2.circle(frame, (nx,ny), 4, (0,0,255), -2)
                if lcx > half :
                    self.detect_drowsiness(frame, face_landmarks, h, w, half)
        
        frame = self.detect_phone(frame, half)
        
        return frame

    def detect_drowsiness(self, frame, face_landmarks, h, w, half):
        # Eye aspect ratio calculation
        left_eye = [face_landmarks.landmark[33], face_landmarks.landmark[160], face_landmarks.landmark[158],
                    face_landmarks.landmark[133], face_landmarks.landmark[153], face_landmarks.landmark[144]]
        right_eye = [face_landmarks.landmark[362], face_landmarks.landmark[385], face_landmarks.landmark[387],
                     face_landmarks.landmark[263], face_landmarks.landmark[373], face_landmarks.landmark[380]]
        
        left_ear = self.eye_aspect_ratio(left_eye, w, h)
        right_ear = self.eye_aspect_ratio(right_eye, w, h)
        mean_ear = (left_ear + right_ear) / 2.0
        
        if mean_ear < self.ear_threshold:
            self.eyeClose += 1
        else:
            self.eyeClose = 0
        
        cv2.putText(frame, f"EAR: {mean_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, f"Eye Close: {self.eyeClose}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.eyeClose >= self.max_frame_ec:
            cv2.putText(frame, "Drowsiness!!!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.status_label.text = "Status: Drowsy!"
            self.play_beep()
        else:
            cv2.putText(frame, "Active:)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.status_label.text = "Status: Active"
        
        # Head pose estimation
        self.head_pose(frame, face_landmarks, h, w)

    def eye_aspect_ratio(self, eye, w, h):
        points = [(landmark.x * w, landmark.y * h) for landmark in eye]
        vert_dist1 = dist.euclidean(points[1], points[5])
        vert_dist2 = dist.euclidean(points[2], points[4])
        horz_dist = dist.euclidean(points[0], points[3])
        ear = (vert_dist1 + vert_dist2) / (2.0 * horz_dist)
        return ear

    def head_pose(self, frame, face_landmarks, h, w):
        image_points = np.array([
            (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),
            (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h),
            (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),
            (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h),
            (face_landmarks.landmark[78].x * w, face_landmarks.landmark[78].y * h),
            (face_landmarks.landmark[308].x * w, face_landmarks.landmark[308].y * h)
        ], dtype="double")
        
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        dist_coeffs = np.zeros((4,1))
        
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs)
        
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),
                                                         rotation_vector, translation_vector,
                                                         camera_matrix, dist_coeffs)
        
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
        cv2.line(frame, p1, p2, (255,0,0), 2)
        
        # Calculate Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_mat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        yaw = euler_angles[1]
        
        if yaw <= -20:
            text = "Right"
            self.dcounter += 1
        elif yaw >= 20:
            text = "Left"
            self.dcounter += 1
        else:
            text = "Straight"
            self.dcounter = 0
        
        cv2.putText(frame, f"Head: {text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if self.dcounter > self.max_frame_direction:
            cv2.putText(frame, "Distracted!!!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.status_label.text = "Status: Distracted!"
            self.play_beep()
        else:
            cv2.putText(frame, "Focused", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def detect_phone(self, frame, half):
        results = self.model(frame, classes=self.class_id, conf=0.3)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                
                if self.operators[self.direction](int(x1), half):
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 200), 2)
                    
                    # Display class name and confidence
                    label = f"Phone: {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)
                    
                    # Status update and beep
                    status_text = f"Phone Detected (Conf: {conf:.2f})"
                    cv2.putText(frame, status_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.status_label.text = f"Status: {status_text}"
                    self.play_beep()
        
        return frame

    def play_beep(self):
        current_time = time.time()
        if current_time - self.last_beep_time > 1:  # Play beep at most once per second
            if self.beep_sound:
                self.beep_sound.play()
            self.last_beep_time = current_time

    def on_stop(self):
        # Release the camera when the app is closed
        self.capture.release()

if __name__ == '__main__':
    DrowsinessDetectionApp().run()