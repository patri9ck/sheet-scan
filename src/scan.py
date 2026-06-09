import cv2


from src.util import show_image

import numpy as np




def main():
    image_path = "../data/PXL_20260527_093456545.jpg"

    original = cv2.imread(image_path)

    show_image(original, "original")

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    edges = cv2.Canny(blurred, 20, 100)

    show_image(edges, "edges")

    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)

    show_image(dilated, "dilated")

    lines = cv2.HoughLinesP(dilated, rho=1, theta=np.pi / 180,
                            threshold=200, minLineLength=100, maxLineGap=50)

    hough = np.zeros_like(dilated)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            cv2.line(hough, (x1, y1), (x2, y2), color=255, thickness=2)

    show_image(hough, "hough")

    closed = cv2.morphologyEx(hough, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)))

    show_image(closed, "closed")



main()