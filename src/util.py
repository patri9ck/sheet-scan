import cv2
from matplotlib import pyplot as plt


def show_image(image, title="Image"):
    if len(image.shape) == 2:
        plt.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation="nearest")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    plt.show()