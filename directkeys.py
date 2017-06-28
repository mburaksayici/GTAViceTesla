import ctypes
import time
import numpy as np

SendInput = ctypes.windll.user32.SendInput
import time

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def w():
    
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    PressKey(W)

def a():
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    PressKey(A)
    

def s():
    
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    PressKey(S)
    

def d():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)

def nokey():
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    
def wa():
    ReleaseKey(D)
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    PressKey(W)
    PressKey(A)
    
    
def wd():
    
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    PressKey(W)
    PressKey(D)
    

def sa():
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    PressKey(S)
    PressKey(A)
    

def sd():
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    PressKey(S)
    PressKey(D)



"""""
def process_img(image,lower_yellow,upper_yellow,lower_white,upper_white):
    original_image = image

    img = cv2.convertScaleAbs(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask2 = cv2.inRange(hsv, lower_white, upper_white)
    combined_mask = cv2.bitwise_or(mask, mask2)
    processed_img = cv2.bitwise_and(image, image, mask = combined_mask)


    # convert to gray
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img = cv2.Canny(processed_img, threshold1=250, threshold2=300)

    processed_img = cv2.GaussianBlur(processed_img, (5,5), 1)

    vertices = np.array([[0, 480], [0, 300], [80, 210], [560, 210],[640,300],[640,480]], np.int32)


    processed_img = roi(processed_img, [vertices])

    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    #                                     rho   theta   thresh  min length, max gap:

    m1 = 0
    m2 = 0
    lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180, 0, 15)



    try:
        l1, l2, m1, m2 = draw_lanes(original_image, lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0, 255, 0], 30)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0, 255, 0], 30)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 0, 0], 30)
                print("2.ci")
                cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0, 255, 0], 30)
                print("oha")
            except Exception as e:
                print(str(e))
    except Exception as e:
        pass
    print(m1,"  ",m2)
    return processed_img,original_image, m1, m2
"""



""""
lower_yellow = np.array([13, 89, 103])
upper_yellow = np.array([150, 200, 175])

lower_white = np.array([20, 75, 100])
upper_white = np.array([30, 100, 120])

def process_img(image,lower_yellow,upper_yellow,lower_white,upper_white):
    original_image = image

    img = cv2.convertScaleAbs(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask2 = cv2.inRange(hsv, lower_white, upper_white)
    combined_mask = cv2.bitwise_or(mask, mask2)
    processed_img = cv2.bitwise_and(image, image, mask = combined_mask)


    # convert to gray
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2HLS)
    # edge detection
    processed_img = cv2.Canny(processed_img, threshold1=250, threshold2=300)

    processed_img = cv2.GaussianBlur(processed_img, (5,5), 1)

    vertices = np.array([[0, 480], [0, 300], [80, 210], [560, 210],[640,300],[640,480]], np.int32)


    processed_img = roi(processed_img, [vertices])

    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    #                                     rho   theta   thresh  min length, max gap:


    lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 112, 150, 15)



    try:
        l1, l2, m1, m2 = draw_lanes(original_image, lines)

        print("m1 ve m2 try in altındaki", m1,m2)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0, 255, 0], 30)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0, 255, 0], 3)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(original_image, (coords[0], coords[1]), (coords[2], coords[3]), [255, 0, 0], 3)
                print("2.ci")
                cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0, 255, 0], 30)
                print("oha")
            except Exception as e:
                print(str(e))
    except Exception as e:
        pass
    return processed_img,original_image,
"""