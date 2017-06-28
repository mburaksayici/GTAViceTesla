""
import numpy as np
from PIL import ImageGrab
import cv2
import time
from numpy import ones, vstack
from numpy.linalg import lstsq
import directkeys
from statistics import mean
from collections import deque
import grabscreen





def select_rgb_white_yellow(image):
    # white color mask

    lower_yellow = np.array([13, 89, 103])
    upper_yellow = np.array([150, 200, 175])
    lower_white = np.array([20, 75, 100])
    upper_white = np.array([30, 100, 120])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    # yellow color mask
    yellow_mask = cv2.inRange(image, lower_yellow, upper_yellow)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be positive and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])  # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)


def select_region(image):
    """""
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices

    mask = np.zeros_like(image)
    vertices = np.array([[0, 480], [0, 300], [80, 210], [560, 210],[640,300],[640,480]],np.int32)
    cv2.fillPoly(mask, [vertices], 255)
    masked = cv2.bitwise_and(image,  mask)
    return masked


def hough_lines(image):
    """x"
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """

    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)


    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue  # ignore a vertical line
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                if slope < 0:  # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))

    # add more weight to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)




def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0]  # bottom of the image
    y2 = y1 * 0.6  # slightly lower than the middle

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line





def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)



class LaneDetector:
    QUEUE_LENGTH=100
    def __init__(self):
        self.left_lines  = deque(maxlen=100)
        self.right_lines = deque(maxlen=100)

    def process(self, image):
        regions=select_region(image)
        white_yellow = select_white_yellow(regions)
        gray         = convert_gray_scale(white_yellow)
        smooth_gray  = apply_smoothing(gray)
        edges        = detect_edges(smooth_gray)
        lines        = hough_lines(edges)
        left_line, right_line = lane_lines(image, lines)

        def mean_line(line, lines):
            if line is not None:
                lines.append(line)

            if len(lines)>0:
                line = np.mean(lines, axis=0, dtype=np.int32)
                line = tuple(map(tuple, line)) # make sure it's tuples not numpy array for cv2.line to work
            return line

        left_line  = mean_line(left_line,  self.left_lines)
        right_line = mean_line(right_line, self.right_lines)

        return left_line,right_line



screen = grabscreen.grab_screen(region=(0, 40, 640, 480))
def process_video(screen):
    detector = LaneDetector()



    while True:
        screen = grabscreen.grab_screen(region=(0, 40, 640, 480))
        last_time = time.time()
        print('Loop took {} seconds'.format(time.time() - last_time))
        detector.process(screen)
        cv2.imshow('window', screen)


        # cv2.imshow('window2', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


process_video(screen)




"""


def main():
    for i in list(range(1))[::-1]:
        print(i+1)
        time.sleep(1)
    last_time = time.time()
    while True:
        screen = grabscreen.grab_screen(region=(0, 40, 640, 480))
        print('Frame took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        processed_img, original_image = process(screen)
        cv2.imshow('window', processed_img)
        cv2.imshow('window2', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        # cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))

"""

""""