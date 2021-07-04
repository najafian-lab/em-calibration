from calibration.grid import GridProcessor
import os
import random
import numpy as np
import threading
import queue
import cv2

IMAGE_FOLDER = 'C:\\Users\\smerk\\Downloads\\images'
IMAGES = os.listdir(IMAGE_FOLDER)
SAMPLES = 8
IMAGES_SAMPLE = random.sample(IMAGES, SAMPLES)
IMAGES_FULL = [os.path.join(IMAGE_FOLDER, image) for image in IMAGES]
IM_SIZE = (400, 400)
THREADS = 5

image_cache = {}
settings = {}

# let's process them
def process_image(item: str, o_q: queue.Queue):
    global settings, image_cache
    if item in image_cache:
        image = image_cache[item].copy()
    else:
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_cache[item] = image

    process = GridProcessor(image, 1, settings)
    process.preprocess_image()
    image = process.processed
    name = os.path.basename(item)
    o_q.put((name, cv2.resize(image, IM_SIZE, interpolation=cv2.INTER_LANCZOS4)))


def process_image_queue(q: queue.Queue, o_q: queue.Queue):
    while True:
        item = q.get()
        if item is None:
            break
        process_image(item, o_q)
        q.task_done()


def process_images():
    global IMAGES_FULL
    threads = []
    in_q = queue.Queue()
    out_q = queue.Queue()
    for _ in range(THREADS):
        thread = threading.Thread(target=process_image_queue, args=(in_q, out_q))
        thread.start()
        threads.append(thread)
    
    # push into queue
    for im in IMAGES_FULL:
        in_q.put(im)

    # end the queues
    for _ in range(THREADS * 3):
        in_q.put(None)

    # join the threads
    for thread in threads:
        thread.join()

    # display the output images
    while True:
        try:
            (name, image) = out_q.get_nowait()
            cv2.imshow(name, image)
        except queue.Empty:
            break

# controls
def change_kernel(val):
    global settings
    k = 2*val + 1
    settings.update({
        'kernel': int(k)
    })
    process_images()


def change_alpha(val):
    global settings
    settings.update({
        'contrast_alpha': int(val)
    })
    process_images()


def change_canny_low(val):
    global settings
    settings.update({
        'canny_low': int(val)
    })
    process_images()


def change_canny_high(val):
    global settings
    settings.update({
        'canny_high': int(val)
    })
    process_images()


# show all of the images
try:
    blank = np.zeros((10, 400), np.uint8)
    cv2.imshow('control', blank)
    process_images()
    cv2.createTrackbar('kernel', 'control', 0, 10, change_kernel)
    cv2.createTrackbar('alpha', 'control', 1, 110, change_alpha)
    cv2.createTrackbar('canny_low', 'control', 1, 250, change_alpha)
    cv2.createTrackbar('canny_high', 'control', 1, 250, change_alpha)
    cv2.waitKey(0)
finally:
    cv2.destroyAllWindows()