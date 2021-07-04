import cv2
import numpy as np
from numpy.core.fromnumeric import mean
import math
from sklearn.mixture import GaussianMixture
from calibration.hough_bundler import HoughBundler
import random
import statistics
import cProfile


class GridProcessor:
    def __init__(self, img_path, pixel):
        self.img_path = img_path
        self.pixel = pixel
    
    ## ==== Helper function Section ====

    # function to get slope of hough lines
    def getSlopeOfLine(self, line):
        xDis = line[0][2] - line[0][0]

        if (xDis == 0):
            return None

        return (line[0][3] - line[0][1]) / xDis

    # function to find number of elements in certain range
    def findFrequency(self, slopes, limit):
        count = 0
        for slope in slopes:
            if slope > limit[0] and slope < limit[1]:
                count += 1
        return count

    # function to find horizontal distance between lines
    def findAverageDistance(self, img):
        y_pos = random.sample(range(math.floor(img.shape[0]*1/4), math.floor(img.shape[0]*3/4)), 30)
        distance_count = 0
        distances = []
        ready_stop = False
        for y in y_pos:
            distance_count = 0
            ready_stop = False

            # iterate util find first white  
            length = len(img[0])
            index = 0
            while index < length and all(img[y][index] == [0, 0, 0]) :
                index += 1

            for x in range(index, length - 1):        
                
                if all(img[y][x] == [255, 255, 255]):
                    if ready_stop:
                        distances.append(distance_count)
                        distance_count = 0
                        ready_stop = False
                    else:
                        distance_count += 1
                else:
                    distance_count += 1
                    ready_stop = True

        distances.sort(key=float)
        length = len(distances)

        print('STD', np.std(distances))

        if (length > 200) :
            distances = distances[math.floor(length/2.6) : -math.floor(length/2.6)]
            return sum(distances) / len(distances)
        return None
    
    def produce_lines_mean(self, img, alpha, beta):

        # STEP.1 ==== Increase contrast and brightness of original picture ==== 

        image = img
        if image is None:
            print('Could not open or find the image: ', self.img_path)
            exit(0)

        ## STEP.2 ==== Read in the new picture(higher contrast and bright) and find hough lines ====

        # img = new_image
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

        new_image = np.zeros(image.shape, image.dtype)

        blur_gray = cv2.equalizeHist(blur_gray)
        blur_gray = np.clip((alpha+2)*blur_gray + beta, a_min=0, a_max=255).astype(np.uint8)


        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        cv2.imshow('edges', edges)


        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 100  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)


        ### STEP.3 ==== Sort the slopes properly and remove the extreme values from it

        slopes = []
        lines_new = []
        for line in lines:
            slope = self.getSlopeOfLine(line)
            if slope:
                slopes.append(self.getSlopeOfLine(line))
                lines_new.append(line)


        slopes.sort(key = float)
        length = len(slopes)
        slopes = slopes[math.floor(length/10) : -math.floor(length/10)]




        ### STEP.4 ==== Fit a multiple guassian distribution to slopes to get two peaks (two major slopes) ==== 

        S = np.array(slopes)
        gm = GaussianMixture(n_components=5, random_state=0).fit(S.reshape(-1, 1))
        best_means = ()
        best_approah = 0
        means = [gm.means_[0][0],gm.means_[1][0],gm.means_[2][0],gm.means_[3][0],gm.means_[4][0]]

        for m1 in means:
            for m2 in means:
                if abs(m1 * m2 + 1) < abs(best_approah + 1):
                    best_approah = m1 * m2
                    best_means = (m1, m2)

        if best_means[0] >  best_means[1]: best_means = best_means[::-1]

        # Use a window of (-0.4, + 0.4) to scan center from (best_mean - 0.1, best mean + 0.1) to find the best window
        best_1 = 0
        best_2 = 0
        for i in range(1, 7):
            center_1 = best_means[0] - (-0.3 + (i - 1) * 0.1)
            center_2 = best_means[1] - (-0.3 + (i - 1) * 0.1)
            freq_1 = self.findFrequency(slopes, (center_1 - 0.4, center_1 + 0.4))
            freq_2 = self.findFrequency(slopes, (center_2 - 0.4, center_2 + 0.4))
            # print(slopes[:10])
            if (freq_1 > best_1):
                best_means = (center_1, best_means[1])
                best_1 = freq_1
            if (freq_2 > best_2):
                best_means = (best_means[0], center_2)
                best_2 = freq_2
        
        # print(best_means)
        
        return(lines_new, best_means)


    def calculate_distance(self, alpha, beta):
        image = cv2.imread(self.img_path)
        black_image = np.zeros((image.shape[0], image.shape[1])) # Create a black image to draw lines on
        lines_new, best_means = self.produce_lines_mean(image, alpha, beta)

        index = 0
        if abs(best_means[0]) < abs(best_means[1]): index = 1

        if (abs(best_means[index]) > 8):
            (h, w) = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY), 20, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            lines_new, best_means = self.produce_lines_mean(rotated, alpha, beta)
 

        ### STEP.5 ==== Draw lines on the black image with slopes in [peak-0.4,peak+0.4] range ==== 

        points = []
        total_slope = 0
        count = 0
        index = 0
        if abs(best_means[0]) < abs(best_means[1]): index = 1


        # for p1, p2 in lines_new:
        #     # sl = self.getSlopeOfLine(line)
        #     # if (sl > best_means[index] - 0.4 and sl < best_means[index] + 0.4):
        #     #     total_slope += sl
        #     #     count += 1
        #     # for p1, p2 in line:
        #     x1, y1 = int(p1[0]), int(p2[1])
        #     x2, y2 = int(p2[0]), int(p2[1])
        #     points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
        #     cv2.line(black_image, (x1, y1), (x2, y2), (255, 0, 0), 8)

        for line in lines_new:
            sl = self.getSlopeOfLine(line)
            # index = 0
            # if abs(best_means[0]) < abs(best_means[1]): index = 1
            if (sl > best_means[index] - 0.3 and sl < best_means[index] + 0.3):
                total_slope += sl
                count += 1
                for x1, y1, x2, y2 in line:
                    points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                    cv2.line(black_image, (x1, y1), (x2, y2), (255, 0, 0), 4)

        # cv2.imshow('test', black_image)
        # cv2.waitKey(0)
        # exit(1)

        avergae_slope = total_slope / count

        ### STEP.6 ==== Calculate horizontal pixel distances between lines and then figure out grid size ==== 
        distance = self.findAverageDistance(black_image)
        if distance:
            return (self.pixel / abs(distance * avergae_slope / math.sqrt(1 + avergae_slope**2)))

        return None

    def get_distance(self):
        list_1 = []
        list_2 = []
        for i in range(0, 4):
            x = self.calculate_distance(2.9, 40)
            y = self.calculate_distance(2.3, 10)
            if x and y:
                list_1.append(x)
                list_2.append(y)
            elif not x and not y:
                return 1000
            elif not x:
                list_2.append(y)
                return mean(list_2)
            else:
                list_1.append(x)
                return mean(list_1)
        if statistics.stdev(list_1) < statistics.stdev(list_2):
            return mean(list_1)
        else:
            return mean(list_2)




processor = GridProcessor('C:\\Users\\smerk\\UW\\Najafian Lab - Lab Najafian\\Foot Process Workspace\\report\\Fabry FPW-Test\\08-0050-20190328T205237Z-001\\08-0050\\08-0050 bi-2\\08_01615.tif', 463)
print(processor.get_distance())