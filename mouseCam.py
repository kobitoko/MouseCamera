import cv2
import mediapipe as mp
import time
import mouse

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# camera input choices later
capture = cv2.VideoCapture(3)
try:
    # Taken from https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer/python
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
        running_mode=VisionRunningMode.VIDEO)
    with GestureRecognizer.create_from_options(options) as recognizer:
        print("Hi")
        while(True):
            ret, frame = capture.read()
            # flip image horizontally
            cv2.flip(frame,1)
            #rgbFromBgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            currentTime = round(time.time() * 1000)
            #recognizer.recognize_async(mpImage, currentTime)
            # video
            result = recognizer.recognize_for_video(mpImage, currentTime)
            if result and result.gestures and len(result.gestures)>0 and len(result.gestures[0]) > 0:
                print(result.gestures[0][0].category_name)
            cv2.imshow('Test hand', frame)
            # stop if ESC is pressed.
            if cv2.waitKey(33) == 27:
                break
finally:
    print("Good bye.")
    cv2.destroyAllWindows()
    capture.release()
