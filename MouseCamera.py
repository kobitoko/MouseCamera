import cv2
import mediapipe as mp
import time
import pyautogui

class MouseCamera:
    def __init__(self, webcamIndex = 0):
        # openCV
        self.camera = cv2.VideoCapture(webcamIndex)
        # mediapipe stuff
        self.base_options = mp.tasks.BaseOptions
        self.gesture_recognizer = mp.tasks.vision.GestureRecognizer
        self.gesture_recognizer_options = mp.tasks.vision.GestureRecognizerOptions
        self.vision_running_mode = mp.tasks.vision.RunningMode
        self.options = self.gesture_recognizer_options(
                num_hands = 1,
                min_hand_detection_confidence = 0.5,
                min_tracking_confidence = 0.5,
                base_options = self.base_options(model_asset_path='gesture_recognizer.task'),
                running_mode = self.vision_running_mode.VIDEO)
        # pyautogui only supports primary monitor for now, it isn't reliable for the screen of a second monitor.
        self.screen_width, self.screen_height = pyautogui.size()
        # Failsafe is thumbs down which causes this to quit.
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.001

        self.running = True
        self.x = 0
        self.y = 0
        self.last_x = 0
        self.last_y = 0
        self.left_mouse_active = False
        self.right_mouse_active = False
        self.left_mouse_down = False
        self.right_mouse_down = False
        self.scroll_active = False
        self.mouse_move = False
        self.show_webcam = False
        # mouse move speed
        self.multiplier = 1.5
        # scroll speed
        self.scroll_multiplier = 2
        # pixels radius movement should ignore
        self.threshold_move = 10

    def set_webcam_index(self, index = 0):
        self.camera = cv2.VideoCapture(index)

    def process_result(self, result):
        # result structure https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer/python#handle_and_display_results
        results_exists = result and result.gestures and len(result.gestures) > 0 and len(result.gestures[0]) > 0
        if results_exists:
            # The canned gestures are category_name ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
            # ILoveYou is apparently thumb, index, and pinkie extended.
            #print(result.gestures[0][0].category_name, result.gestures[0][0].score)
            match result.gestures[0][0].category_name:
                case "Closed_Fist":
                    self.mouse_move = True
                    self.scroll_active = False
                    self.reset_mouse_buttons()
                case "Pointing_Up":
                    self.left_mouse_active = True
                    self.scroll_active = False
                case "Victory":
                    self.right_mouse_active = True
                    self.scroll_active = False
                case "Thumb_Up":
                    self.scroll_active = True
                case "Open_Palm" | "None":
                    self.reset_mouse_states()
                case "Thumb_Down":
                    if result.gestures[0][0].score > 0.85:
                        self.reset_mouse_states()
                        self.running = False
                case _:
                    print("process_result defaulted with case:", result.gestures[0][0].category_name)
            if self.any_mouse_state_active():
                # Value from https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer#hand_landmark_model_bundle
                wrist = result.hand_landmarks[0][0]
                x = wrist.x
                y = wrist.y
                if self.last_x == -1 or self.last_y == -1:
                    # it was last reset, we don't count those last values
                    self.last_x = x
                    self.last_y = y
                else:
                    self.last_x = self.x
                    self.last_y = self.y
                self.x = x
                self.y = y
        else:
            self.reset_mouse_states()

    def any_mouse_state_active(self):
        return self.mouse_move or self.left_mouse_active or self.right_mouse_active or self.scroll_active

    def reset_mouse_states(self):
        self.reset_mouse_buttons()
        self.mouse_move = False
        self.scroll_active = False
        self.last_x = -1
        self.last_y = -1

    def reset_mouse_buttons(self):
        self.left_mouse_active = False
        self.right_mouse_active = False

    def perform_actions(self):
        if self.left_mouse_active:
            if not self.left_mouse_down:
                pyautogui.mouseDown(button="left")
                self.left_mouse_down = True
        else:
            if self.left_mouse_down:
                pyautogui.mouseUp(button="left")
                self.left_mouse_down = False
        if self.right_mouse_active:
            if not self.right_mouse_down:
                pyautogui.mouseDown(button="right")
                self.right_mouse_down = True
        else:
            if self.right_mouse_down:
                pyautogui.mouseUp(button="right")
                self.right_mouse_down = False
        if self.any_mouse_state_active():
            diff_x = self.x - self.last_x
            diff_y = self.y - self.last_y
            move_x = diff_x * self.screen_width * self.multiplier
            move_y = diff_y * self.screen_height * self.multiplier
            if abs(move_x) < self.threshold_move:
                move_x = 0
            if abs(move_y) < self.threshold_move:
                move_y = 0
            if self.scroll_active:
                # Takes only int, otherwise it crashes.
                scroll_clicks = int(round(move_y * self.scroll_multiplier))
                pyautogui.scroll(scroll_clicks)
            else:
                pyautogui.move(move_x, move_y)

    def start(self):
        try:
            # Taken from https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer/python
            with self.gesture_recognizer.create_from_options(self.options) as recognizer:
                print("Starting")
                while(self.running):
                    ret, frame = self.camera.read()
                    # flip image horizontally
                    frame = cv2.flip(frame,1)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    current_time_ms = round(time.time() * 1000)
                    result = recognizer.recognize_for_video(mp_image, current_time_ms)
                    self.process_result(result)
                    self.perform_actions()
                    if self.show_webcam:
                        cv2.imshow('Webcam View', frame)
                        # run for 16ms (16ms = 60fps) and stop if ESC is pressed.
                        if cv2.waitKey(16) == 27:
                            break
        finally:
            print("Good bye.")
            cv2.destroyAllWindows()
            self.camera.release()

if __name__ == "__main__":
    #DEBUG, my camera is 3
    app = MouseCamera(3)
    app.start()