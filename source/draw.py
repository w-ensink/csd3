import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Draw the results on a frame overlay
def centers(arr, bg, dim, spd, freeze):
    overlay = Image.new('RGBA', dim, '#00000011')
    canvas = ImageDraw.Draw(overlay)

    # Draw square at center coordinate
    for p in arr:
        canvas.polygon([(p[0] - 5, p[1] - 5), (p[0] - 5, p[1] + 5), (p[0] + 5, p[1] + 5), (p[0] + 5, p[1] - 5)],
                       'yellow')

    fnt = ImageFont.truetype("../../_assets/arial.ttf", 20)
    canvas.text((10, 10), str(round(spd, 4)), font=fnt, fill=(255, 255, 255, 255))
    if freeze:
        canvas.text((10, 30), "freeze", font=fnt, fill=(255, 255, 255, 255))

    overlay = np.array(overlay)[:, :, ::-1].copy()
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
    return cv2.addWeighted(bg, 0.7, overlay, 0.3, 0)


def generate_square(center, radius=5):
    return [(int(center[0][0] - radius), int(center[0][1] - radius)), (int(center[0][0] + radius), int(center[0][1] + radius))]