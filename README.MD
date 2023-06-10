## Camera Mouse

Using mediapipe's hand gestures/hand tracking to control the mouse: https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer  
PyAutoGUI to perform the actual mouse actions: https://pyautogui.readthedocs.io/en/latest/index.html  
OpenCV to grab the webcam: https://pypi.org/project/opencv-python/  

- Fist to move mouse.
- Index Finger Pointing up: Left Mouse Click & Hold / Left Mouse Down.
- Index and Middle Finger Pointing up/Victory sign: Right Mouse Click & Hold / Right Mouse Down.
- Thumbs up and move: scrolling in the moving direction (vertical only).
- Thumbs down: quit the application.
- Open palm or anything else: reset state.

Uses Python 3.10: https://www.python.org/downloads/release/python-31011/  
Create virtual environment with python 3.10: `py -3.10 -m venv .\venv`  
To activate environment: `venv\Scripts\activate`  
To install dependencies: `pip install -r requirements.txt`  
Run it: `py MouseCamera.py`  
