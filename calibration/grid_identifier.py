import cv2
import numpy as np


def is_grid(path):
    # img = cv2.imread("images/15-0079.png", 0)
    img = cv2.imread(path,0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    ret, magnitude_spectrum = cv2.threshold(magnitude_spectrum, 255, 255, cv2.THRESH_BINARY)


    cnts = cv2.findContours(magnitude_spectrum.astype(np.uint8), cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE)[-2]

    xareas = []
    for cnt in cnts:
        if cv2.contourArea(cnt) > 9:
            xareas.append(cv2.contourArea(cnt))
    xareas.sort(reverse=True)
    if len(xareas) < 8 or xareas[0] > 1000 or xareas[0] / xareas[1] > 15:
        return False
    return True
    