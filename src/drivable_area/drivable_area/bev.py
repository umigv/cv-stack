import cv2
import numpy as np
from math import radians, cos

class CameraProperties(object):
    functional_limit = radians(70.0)
    def __init__(self, height, fov_vert, fov_horz, cameraTilt):
        self.height = float(height)
        self.fov_vert = radians(float(fov_vert))
        self.fov_horz = radians(float(fov_horz))
        self.cameraTilt = radians(float(cameraTilt))
        self.bird_src_quad = None
        self.bird_dst_quad = None
        self.matrix = None
        self.maxHeight = None
        self.maxWidth = None
        self.minIndex = None

    def src_quad(self, rows, columns):
        if self.bird_src_quad is None:
            self.bird_src_quad = np.array([[0, rows - 1], [columns - 1, rows - 1], [0, 0], [columns - 1, 0]], dtype = 'float32')
        return self.bird_src_quad

    def dst_quad(self, rows, columns, min_angle, max_angle):
        if self.bird_dst_quad is None:
            fov_offset = self.cameraTilt - self.fov_vert/2.0
            bottom_over_top = cos(max_angle + fov_offset)/cos(min_angle + fov_offset)
            bottom_width = columns*bottom_over_top
            blackEdge_width = (columns - bottom_width)/2
            leftX = blackEdge_width
            rightX = leftX + bottom_width
            self.bird_dst_quad = np.array([[leftX, rows], [rightX, rows], [0, 0], [columns, 0]], dtype = 'float32')
        return self.bird_dst_quad

    def reset(self):
        self.bird_src_quad = None
        self.bird_dst_quad = None
        self.matrix = None
        self.maxHeight = None
        self.maxWidth = None
        self.minIndex = None

    def compute_min_index(self, rows, max_angle):
        self.minIndex = int(rows*(1.0 - max_angle/self.fov_vert))
        return self.minIndex

    def compute_max_angle(self):
        return min(CameraProperties.functional_limit - self.cameraTilt + self.fov_vert/2.0, self.fov_vert)

def getBirdView(image, cp):
    if (cp.matrix is None):
        rows, columns = image.shape[:2]
        if columns == 1280:
            columns = 1344
        if rows == 720:
            rows = 752
        min_angle = 0.0
        max_angle = cp.compute_max_angle()
        min_index = cp.compute_min_index(rows, max_angle)
        image = image[min_index:, :]
        rows = image.shape[0]

        src_quad = cp.src_quad(rows, columns)
        dst_quad = cp.dst_quad(rows, columns, min_angle, max_angle)
        return perspective(image, src_quad, dst_quad, cp)
    else:
        image = image[cp.minIndex:, :]
        return cv2.warpPerspective(image, cp.matrix, (cp.maxWidth, cp.maxHeight))

def perspective(image, src_quad, dst_quad, cp):
    bottomLeft, bottomRight, topLeft, topRight = dst_quad
    widthA = topRight[0] - topLeft[0]
    widthB = bottomRight[0] - bottomLeft[0]
    maxWidth1 = max(widthA, widthB)
    heightA = bottomLeft[1] - topLeft[1]
    heightB = bottomRight[1] - topRight[1]
    maxHeight1 = max(heightA, heightB)

    matrix1 = cv2.getPerspectiveTransform(src_quad, dst_quad)
    cp.matrix = matrix1
    cp.maxWidth = int(maxWidth1)
    cp.maxHeight = int(maxHeight1)

    warped = cv2.warpPerspective(image, matrix1, (cp.maxWidth, cp.maxHeight))

    mask = np.full((cp.maxHeight, cp.maxWidth), -1, dtype=np.float32)

    trapezoid = np.array([bottomLeft, bottomRight, topRight, topLeft], dtype = 'int32')
    cv2.fillConvexPoly(mask, trapezoid, 0)

    warped += mask

    return warped