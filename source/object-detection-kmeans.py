import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# feature detection
# - background subtraction
# - background = black, moving object = white
# - count black & white pixels to set percentage

# cap = cv2.VideoCapture(2)
background_subtraction = cv2.createBackgroundSubtractorKNN()
type = "cam"
running = True
w = 480
h = 320

cap = cv2.VideoCapture(2)
vid = cv2.VideoCapture('../../_assets/WIN_20210918_10_58_50_Pro.mp4')


def closing(image):
    kernelSizes = [(3, 3), (5, 5), (7, 7)]

    kn = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSizes[2])
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kn)


def preprocess(image):
    mod = closing(image)
    c = cv2.cvtColor(mod, cv2.COLOR_RGB2RGBA)
    Z2 = cv2.findNonZero(mod)

    if Z2 is not None:
        Z2 = np.squeeze(Z2, axis=1)
        Z2 = Z2.reshape((-1, 2))
        return c, np.float32(Z2)
    else:
        return c, None


def draw_centers(arr, bg):
    overlay = Image.new('RGBA', (w, h), '#ff000066')
    canvas = ImageDraw.Draw(overlay)

    # Draw square at center coordinate
    for p in arr:
        canvas.polygon([(p[0] - 5, p[1] - 5), (p[0] - 5, p[1] + 5), (p[0] + 5, p[1] + 5), (p[0] + 5, p[1] - 5)],
                       'yellow')

    overlay = np.array(overlay)[:, :, ::-1].copy()
    return cv2.addWeighted(bg, 0.7, overlay, 0.3, 0)


def kMeans(image):
    # convert to np.float32
    c, Z2 = preprocess(image)

    if Z2 is not None:
        # define criteria and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret2, label, center = cv2.kmeans(Z2, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        return draw_centers(center, c)
    else:
        return image


def prep_video():
    ret, frame = cap.read()

    return frame


def prep_image():
    im = cv2.imread('../../_assets/test.png')
    return im


wait = True
while running:
    if type == 'image':
        img = cv2.cvtColor(prep_image(), cv2.COLOR_RGB2GRAY)
    if type == 'video':
        _, img = vid.read()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = prep_video()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (w, h))
    kernel = np.ones((5, 5), np.uint8)

    foreground_mask = background_subtraction.apply(img)
    img2 = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

    # TRYING OUT KMEANS CLUSTERING
    ol = kMeans(img2)
    # ============================

    # closing = cv2.add(ol, img)
    cv2.imshow('frame', img)
    cv2.imshow('detect', ol)

    while wait:
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            wait = False

    # moving object
    white_pixels = cv2.countNonZero(img)
    print("white pixels: ", white_pixels)

    # background
    black_pixels = np.sum(img == 0)
    print("black pixels: ", black_pixels)

    percentage_white_pixels = (white_pixels / black_pixels) * 100
    print("percentage of white pixels: ", percentage_white_pixels, "\n")

    k = cv2.waitKey(30) & 0xff
    if type == "image" or k == 27:
        running = False

cap.release()
cv2.destroyAllWindows()
