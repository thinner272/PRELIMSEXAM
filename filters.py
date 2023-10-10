import cv2
import numpy as np
class VConvolutionFilter(object):
  def __init__(self, kernel):
    self._kernel = kernel
  def apply(self, src, dst):
    cv2.filter2D(src, -1, self._kernel, dst)
        
class SharpenFilter(VConvolutionFilter):
  def __init__(self):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    VConvolutionFilter.__init__(self, kernel)
    
class FindEdgesFilter(VConvolutionFilter):
  def __init__(self):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    VConvolutionFilter.__init__(self, kernel)
    
class Blurfilter(VConvolutionFilter):
  def __init__(self):
    kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04]])
    VConvolutionFilter.__init__(self, kernel)
    
class EmbossFilter(VConvolutionFilter):
  def __init__(self):
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
    VConvolutionFilter.__init__(self, kernel)
def strokeEdges(src, dst, blurKsize = 7, edgeKsize = 5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize = edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)
class ContourDetection:
    @staticmethod
    def apply(img):
        cv2.pyrDown(img)
        gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY_INV)
        contours, hier = cv2.findContours (thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grant = cv2.drawContours (img, contours, -1, (0,255,0), 2) 
        return img
def ContourFilter(img1):
    gray_img = cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold (cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY_INV)
    contours, hier = cv2.findContours (thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grant = cv2.drawContours (img1, contours, -1, (0,255,0), 2) 
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img1, (x,y), (x+w, y+h), (0, 255, 0), 2)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box) 
        cv2.drawContours (img1, [box], 0, (0, 0, 255), 3)
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int (radius)
        img1 = cv2.circle(img1, center, radius, (0, 255, 0), 2)

class CircleDetection:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        
    def detect_circles(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray, 15)

            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 99,
                                       param1=120, param2=35, minRadius=0,
                                       maxRadius=0)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

            cv2.imshow('cameo', frame)

            if cv2.waitKey(1) == 13:
                break

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Usage in your main program:
if __name__ == "__main__":
    circle_detector = CircleDetection(camera_index=1)
    circle_detector.detect_circles()
    circle_detector.release()