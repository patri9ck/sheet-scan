import argparse
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image(image, title, show):
    if len(image.shape) == 2:
        plt.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation="nearest")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if show:
        plt.show()
    else:
        os.makedirs("results", exist_ok=True)

        plt.savefig("results/" + title + ".png")



def main():
    parser = argparse.ArgumentParser(description="Document segmentation and rotation")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show images"
    )

    args = parser.parse_args()

    original = cv2.imread(args.image)

    show_image(original, "00_original", args.show)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    edges = cv2.Canny(blurred, 20, 100)

    show_image(edges, "01_edges", args.show)

    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)

    show_image(dilated, "02_dilated", args.show)

    inverted = False

    if np.sum(dilated == 255) / dilated.size > 0.4:
        dilated = cv2.bitwise_not(dilated)

        inverted = True

        show_image(dilated, "03_dilation_inverted", args.show)

    lines = cv2.HoughLinesP(dilated, rho=1, theta=np.pi / 180,
                            threshold=200, minLineLength=100, maxLineGap=50)

    hough = np.zeros_like(dilated)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            cv2.line(hough, (x1, y1), (x2, y2), color=255, thickness=2)

    show_image(hough, "04_hough", args.show)

    if not inverted and np.sum(hough == 255) / hough.size > 0.4:
        hough = cv2.bitwise_not(hough)

        show_image(hough, "05_hough_inverted", args.show)

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

    show_image(filtered, "06_filtered", args.show)

    points = np.column_stack(np.where(filtered == 255))[:, ::-1]  # (y,x) → (x,y)

    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype(np.int32)

    filled = np.zeros_like(filtered)

    cv2.fillPoly(filled, [box], color=255)

    show_image(filled, "07_filled", args.show)

    masked = cv2.bitwise_and(blurred, blurred, mask=filled)

    x, y, w, h = cv2.boundingRect(box)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    cropped = masked[y1:y2, x1:x2]

    show_image(cropped, "08_cropped", args.show)

    f_transform = np.fft.fft2(cropped)
    f_shift = np.fft.fftshift(f_transform)

    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    show_image(magnitude_spectrum, "09_magnitude_spectrum", args.show)

    magnitude_scaled = cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    show_image(magnitude_scaled, "10_magnitude_scaled", args.show)

    _, thresh = cv2.threshold(magnitude_scaled, 180, 255, cv2.THRESH_BINARY)

    show_image(thresh, "11_thresh", args.show)

    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 50, minLineLength=10, maxLineGap=20)

    angle = 0.0

    if lines is not None:
        x1, y1, x2, y2 = max(lines, key=lambda l: (l[0][0] - l[0][2]) ** 2 + (l[0][1] - l[0][3]) ** 2)[0]

        hough = np.zeros_like(thresh)

        cv2.line(hough, (x1, y1), (x2, y2), color=255, thickness=2)

        show_image(hough, "12_hough_fft", args.show)

        angle = float(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi - 90)

    center = (x + w // 2, y + h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_box = cv2.warpAffine(filled, rotation_matrix, (width, height))

    x_r, y_r, w_r, h_r = cv2.boundingRect(rotated_box)

    rotation_matrix[0, 2] -= x_r
    rotation_matrix[1, 2] -= y_r

    rotated = cv2.warpAffine(original, rotation_matrix, (w_r, h_r))

    show_image(rotated, "13_rotated", args.show)

    tx_pts = cv2.transform(box.reshape(-1, 1, 2).astype(np.float32), rotation_matrix).reshape(-1, 2)

    s, d = tx_pts.sum(axis=1), np.diff(tx_pts, axis=1).flatten()
    src_pts = np.float32([tx_pts[np.argmin(s)], tx_pts[np.argmin(d)], tx_pts[np.argmax(s)], tx_pts[np.argmax(d)]])
    dst_pts = np.float32([[0, 0], [w_r - 1, 0], [w_r - 1, h_r - 1], [0, h_r - 1]])

    rectified = cv2.warpPerspective(rotated, cv2.getPerspectiveTransform(src_pts, dst_pts), (w_r, h_r))

    show_image(rectified, "14_rectified", args.show)

    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3.0, -0.5],
                       [0, -0.5, 0]])

    sharpened = cv2.filter2D(rectified, -1, kernel)

    show_image(sharpened, "15_sharpened", args.show)


if __name__ == "__main__":
    main()