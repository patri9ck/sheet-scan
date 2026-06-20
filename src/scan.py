import cv2


from src.util import show_image

import numpy as np




def main():
    image_path = "../data/PXL_20260527_093452881.jpg"

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

    if np.sum(hough == 255) / hough.size > 0.4:
        hough = cv2.bitwise_not(hough)

        show_image(hough, "inverted")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hough, connectivity=8, ltype=cv2.CV_32S)

    height, width = hough.shape

    largest_label = -1
    largest_area = 0

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]

        if x <= 0 or y <= 0 or (x + bw) >= width or (y + bh) >= height:
            continue

        area = stats[label, cv2.CC_STAT_AREA]

        if area > largest_area:
            largest_area = area
            largest_label = label

    filtered = np.zeros_like(hough)

    if largest_label != -1:
        filtered[labels == largest_label] = 255

    show_image(filtered, "filtered")

    points = np.column_stack(np.where(filtered == 255))[:, ::-1]  # (y,x) → (x,y)

    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype(np.int32)

    filled = np.zeros_like(filtered)

    cv2.fillPoly(filled, [box], color=255)

    show_image(filled, "filled")

    masked = cv2.bitwise_and(blurred, blurred, mask=filled)

    x, y, w, h = cv2.boundingRect(box)
    cropped = masked[y:y + h, x:x + w]

    show_image(cropped, "cropped")

    f_transform = np.fft.fft2(cropped)
    f_shift = np.fft.fftshift(f_transform)

    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    show_image(magnitude_spectrum, "magnitude_spectrum")

    magnitude_scaled = cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    show_image(magnitude_scaled, "magnitude_scaled")

    _, thresh = cv2.threshold(magnitude_scaled, 160, 255, cv2.THRESH_BINARY)


main()