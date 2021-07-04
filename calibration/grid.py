from typing import List, Tuple
from sklearn.mixture import GaussianMixture
from calibration.util import *
import cv2
import copy
import numpy as np
import threading as th
import queue
import h5py as hf
import math
import pint

# useful
DEBUG = False
DEBUG_IMAGE = False

# default processing settings
GAUSS_KERNEL = 5
THREADS = 4  # number of threads to run 
PERC_STD_90 = 1.45  # a little less than 90% percentile by standard deviation
INVALID_MEASURE = -1000  # must be the same as in util.pyx
UREG = pint.UnitRegistry()  # default registry for default values


class GridProcessor(object):
    def __init__(self, image: (str, np.ndarray, hf.Dataset), grid_size: (float, str, int, pint.Quantity)=None, settings: dict=None):
        """ Handles processing EM Image grids

        Args:
            image (str, ndarray, Dataset): path to single image or data of image
            grid_size (float, int, Quantity): the conversion unit between a single grid example '100nm' square grids or '1m' square grids
        """

        # set initially to invalid
        self.valid = False
        self.dirty = True  # we need to process the image again

        # load the image data
        if isinstance(image, str):
            self.image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, np.ndarray):
            self.image = image.astype(np.uint8)

            if self.image is not None and len(self.image.shape) == 3 and self.image.shape[2] == 3:  # a color image
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        elif isinstance(image, hf.Dataset):
            self.image = np.array(image[:], dtype=np.uint8)
        else:
            raise ValueError('The image type ' + str(image) + ' is not supported')
    
        # make sure dims are correct and image is valid
        if self.image is None:
            raise RuntimeError('Image cannot be None!')
        if len(self.image.shape) != 2:
            raise RuntimeError('Image must have a shape of (rows, cols) not anything else for the grid image')
        elif self.image.shape[0] < 100 or self.image.shape[1] < 100:
            raise RuntimeError('Image has a oddly small shape and probably is not good to find a grid for ' + str(image.shape))

        # update grid size data
        if grid_size is None:
            grid_size = 1
        if isinstance(grid_size, (float, int)):
            self.grid_size = UREG.Quantity(grid_size, 'pixel')
        elif isinstance(grid_size, pint.Quantity):
            self.grid_size = grid_size
        elif isinstance(grid_size, str):
            self.grid_size = UREG.Quantity(grid_size)

        # construct settings
        self.set_settings(settings)
        self.processed = False
        self.valid = True

    def set_settings(self, settings: dict):
        """ Set the grid processing settings (overwriting all of them)

        Args:
            settings (dict): dict of non-default settings to update
        """
        if settings is None:
            self.settings = {}
        else:
            self.settings = settings

        # we're dirty
        self.dirty = True

    def update_settings(self, settings: dict):
        """ Update settings with only the specified keys provided

        Args:
            settings (dict): settings to update the grid processor with
        """
        self.settings.update(settings)
        
        # we're dirty
        self.dirty = True

    def _gs(self, key: str, default: object) -> object:
        """ Get a setting """
        return self.settings.get(key, default)

    def preprocess_image(self):
        """ Preprocess the grid image by adjusting the contrast, blur, and histogram """
        # blur image
        kernel = self._gs('kernel', 9)
        blur_gray = cv2.bilateralFilter(self.image, kernel, 150, 150) # cv2.GaussianBlur(self.image, (kernel, kernel), 0)
        
        # constrast normalize
        blur_gray_eq = cv2.equalizeHist(blur_gray)

        # adjust contrast
        blur_gray_cont = adjust_contrast(blur_gray_eq, self._gs('contrast_alpha', 5.8), self._gs('contrast_beta', 40), self._gs('contrast_mean_shift', False))

        # detect edges
        edges = cv2.Canny(blur_gray_cont, self._gs('canny_low', 11), self._gs('canny_high', 157))

        # if debugging
        if DEBUG_IMAGE:
            cv2.imshow('original', self.image)
            cv2.imshow('blur', blur_gray)
            cv2.imshow('contrast', blur_gray_cont)
            cv2.imshow('edges', edges)
            cv2.waitKey(0)

        # update final ref
        self.processed = edges

    def process_lines(self):
        rho = self._gs('hough_rho', 1)  # distance resolution in pixels of the Hough grid
        theta = self._gs('hough_theta', np.pi / 180)  # angular resolution in radians of the Hough grid
        threshold = self._gs('hough_thresh', 17)  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = self._gs('hough_min_line_length', 100)  # minimum number of pixels making up a line
        max_line_gap = self._gs('hough_max_line_gap', 25)  # maximum gap in pixels between connectable line segments

        # run Hough on edge detected image
        # output "lines" is an array containing endpoints of detected line segments
        self.lines = cv2.HoughLinesP(self.processed, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

        # if debug image let's draw the hough lines
        if DEBUG_IMAGE:
            line_img = np.zeros(self.image.shape[:2], dtype=np.uint8)
            draw_lines(line_img, self.lines, 0, 1000, self._gs('line_width', 4), (255, 255, 255))
            cv2.imshow('hough_lines', line_img)
            cv2.waitKey(8000)


        # make sure it's valid we need at least N lines
        min_lines_required = self._gs('hough_min_lines', 5)
        if len(self.lines) < min_lines_required:
            self.valid = False

            if DEBUG:
                print('Initial hough lines not enough lines')
            return

        # get the slopes of the lines
        self.line_angles = hough_line_angles(self.lines).reshape(-1)

        # get the mixins to get the best means
        n_comp = int(self._gs('gauss_components', 8))  # number of gaussian means to cluster to
        mean_right_tolerance = float(self._gs('mean_angle_tolerance', 0.35))  # tolerance from pi/2 to be from to count these means
        means = []  # keep at least 1 to start
        tries = 0

        while tries < 5:
            gm = GaussianMixture(n_components=n_comp, random_state=0, tol=1e-5).fit(self.line_angles.reshape(-1, 1))
            
            # if we didn't converge let's quit
            if not gm.converged_:
                self.valid = False
                return

            # find those that have right angles
            means = [gm.means_[i][0] for i in range(n_comp)]

            if DEBUG:
                print('Test MEANS', means)

            # let's scan means to reduce components
            passed = True
            for m in means:
                if m == 0.0:  # chances of true mean being zero is so so unlikely we can take that hit
                    n_comp -= 1  # reduce a component for each 0.0
                    passed = False
                    break
                
            # let's not try too many times
            tries += 1

            if DEBUG:
                print('Trying angle compare means', means)

            # get the two best angles for the means
            best_means = None
            best_approah = 10000
            for i, m1 in enumerate(means):
                for j, m2 in enumerate(means):
                    if i == j:  # don't reuse same mean
                        continue

                    # if our means are within tolerance
                    tol = abs((abs(m1) + abs(m2)) - (math.pi / 2.0))
                    if tol < best_approah and tol < mean_right_tolerance:
                        best_approah = tol
                        best_means = (m1, m2)
        
            if best_means is None:
                passed = False

            # if all components used then let's skip
            if passed:
                break

        # couldn't find anything within angle tolerances
        if best_means is None:
            self.valid = False

            if DEBUG:
                print('Did not find means within tolerance', means)

            return

        # save means and best means
        self.means = means
        self.best_means = best_means

        # let's shuffle through nearby means
        resolution = self._gs('angle_scan_resolution', 0.025)  # resolution to scan all of the angles
        window = self._gs('angle_window', 0.15)  # max angle window to scan from the perpindicular best
        self.tolerance = self._gs('tolerance', 0.15)  # angle of tolerance between each frequency scan so window +- tolerance
        self.best_means = scan_angle_frequency_range(self.line_angles, self.best_means[0], self.best_means[1], resolution, window, self.tolerance)

        # check to make sure the means are not the same number
        if self.best_means[0] == self.best_means[1]:
            self.valid = False

            if DEBUG:
                print('Same means... exiting')

        if DEBUG:
            print('Best means', self.best_means)
            print('All means', self.means)

        # make sure there are valid counts
        self.best_freq = [find_frequency(self.line_angles, v - self.tolerance, v + self.tolerance) for v in self.best_means]
        if all([v < min_lines_required for v in self.best_freq]):
            self.valid = False

            if DEBUG:
                print('Not enough lines met', self.best_freq)

        # sort to have the one with fewer mean deviations first
        freqs = [(i, filter_angles(self.line_angles, self.best_means[i] - self.tolerance, self.best_means[i] + self.tolerance)) for i in range(len(self.best_means))]
        mean_stds = [((i, np.std(f)) if len(f) > 0 else (i, 10000)) for i, f in freqs]  # calc the STDs
        mean_stds = sorted(mean_stds, key=lambda x: x[1])
        self.best_means = [self.best_means[ind] for ind, _ in mean_stds]  # get the mean values from the sorted STDs

    def make_line_image(self, with_background: bool=False, line_color: (tuple)=(255, 255, 255), means_index: int=0) -> np.ndarray:
        if with_background:
            self.line_image = self.image.copy()
        else:
            self.line_image = np.zeros(self.image.shape[:2], dtype=np.uint8)

        self.average_angle = draw_lines(self.line_image, self.lines, self.best_means[means_index], self.tolerance, self._gs('line_width', 4), line_color)
        
        if DEBUG_IMAGE:
            if DEBUG:
                print('Image with tolerance %f' % self.tolerance, ' drawing mean ', self.best_means[means_index], ' tolerance ', self.tolerance)
            cv2.imshow('lines', self.line_image)
            cv2.waitKey(8000)

        return self.line_image

    def get_image_distances(self):
        padding = self._gs('scan_padding', 0.23)  # 20% of the image
        measures = self._gs('make_measures', 120)  # make 60 measures
        min_distance = self._gs('min_distance', 5)  # min distance of grid in pixels
        max_distance = self._gs('max_distance', 100)  # max distance of grid in pixels
        distances = find_horz_image_distance(self.line_image, padding, measures, min_distance, max_distance)
        return distances

    def process_image(self, alpha: float=None, beta: float=None, force: bool=False):
        # update settings if applied
        if alpha and alpha != self.settings['contrast_alpha'] :
            self.settings['contrast_alpha'] = alpha
            self.dirty = True
        if beta and beta != self.settings['contrast_beta']:
            self.settings['contrast_beta'] = beta
            self.dirty = True

        # if we're not dirty there is no point in processing again
        if not self.dirty and not force:
            return

        # process the image (checking that it's valid along the way)
        self.preprocess_image()
        
        # get the hough lines and get the best two perpindicular lines
        if self.valid:
            self.process_lines()

        # draw the valid lines on the image
        if self.valid:
            self.make_line_image(line_color=(255, 255, 255), means_index=0)  # first set of lines
            
            # weird result let's try the other way
            if self.average_angle == INVALID_MEASURE:
                self.make_line_image(line_color=(255, 255, 255), means_index=1)  # second set of lines

                # none of the lines are valid :(
                if self.average_angle == INVALID_MEASURE:
                    self.valid = False

        # measure the distances between the lines
        if self.valid:
            distances = self.get_image_distances()
            # print('distances', distances)

            # if we didn't make enough measurements then let's rotate the image
            min_lines = self._gs('min_scan_lines', 5)
            rotate = len(distances) < min_lines
            if rotate:
                self.line_image = rotate_image(self.line_image, 35)  # rotate image 35 degrees to possibly have more scans
                distances = self.get_image_distances()

            if len(distances) < min_lines:
                self.valid = False


            self.distances = distances

    def _process_image_queue(self, input: queue.Queue, output: queue.Queue):
        while True:
            item = input.get()
            if item is None:
                break

            (image, settings) = item
            proc = GridProcessor(image, settings=settings)
            proc.process_image(force=True)
            if proc.valid:
                output.put(proc)
            else:
                output.put(None) # invalid
            input.task_done()

    def process_image_sequence(self):
        # process the sequences of images using the variance settings
        settings_list = self._gs('sequence_settings', [
            {
                'contrast_alpha': 5.8,
                'contrast_beta': 40,
                'canny_low': 10,
                'canny_high': 157
            },
            {
                'contrast_alpha': 2.5,
                'contrast_beta': 10,
                'canny_low': 50,
                'canny_high': 150
            }
        ])

        # make copies of the objects and start the new threads
        threads = []
        ins = queue.Queue()
        outs = queue.Queue()
        for i in range(THREADS):
            thread = th.Thread(target=self._process_image_queue, args=(ins, outs), name='Grid Settings Processor %d' % i)
            thread.start()
            threads.append(thread)
        
        # pass the image settings
        for setting in settings_list:
            ins.put((self.image, setting))
        
        # if debug let's put a waitkey
        if DEBUG:
            cv2.waitKey(0)

        # let's place the nones to kill the threads
        for _ in range(THREADS * 2):
            ins.put(None)
        
        # let's wait for all of them to finish
        for thread in threads:
            thread.join()
        
        # no longer dirty
        self.dirty = False

        # compile the grid objects
        grids = []
        while True:
            try:
                item = outs.get_nowait()

                # add to the grid (if we made measurements on the settings)
                if item is not None and item.valid:
                    grids.append(item)
            except queue.Empty:
                break
        
        # now let's compare the results
        if len(grids) == 0:
            self.valid = False
        else:
            best = None
            best_std = float(self._gs('max_1std', 80.0))  # standard deviation (max deviation for 70 percentile is X pixels)

            # find the best grid based on the variation of measurements
            for grid in grids:
                dist = grid.distances
                std = float(np.std(dist))
                if std < best_std:
                    best = grid
                    best_std = std
            
            # we couldn't find any with low variation
            if best is None:
                self.valid = False

                if DEBUG:
                    print('Did not pass standard deviation of %f' % float(best_std))
            else:
                # copy the useful outputs
                self.distances = grid.distances
                self.lines = grid.lines
                self.line_angles = grid.line_angles
                self.best_means = grid.best_means
                self.best_freq = grid.best_freq
                self.average_angle = grid.average_angle
                
                # now let's compute the distance by the average angle
                mean = np.mean(self.distances)
                lower_tol = mean - (best_std * PERC_STD_90)
                upper_tol = mean + (best_std * PERC_STD_90)
                dist_double = self.distances.astype(np.double)
                distances = filter_angles(dist_double, lower_tol, upper_tol)
                if len(distances) < 5:  # too few measurements
                    distances = dist_double  # let's use all of them
                
                # capture the new average distance and compute the direct distance between the lines
                self.distance = float(np.mean(distances) * abs(np.cos(math.pi - ((math.pi / 2.0) + self.average_angle))))

    def get_grid_distance(self, force: bool=False) -> float:
        """ Gets the grid size by computing the grids from the image

        Args:
            force (bool, optional): Forcibly compute the sequence. Defaults to False.

        Returns:
            float: average distance (using multiple methods) with lowest standard deviation image in pixels
        """
        if self.dirty or force:
            self.dirty = True
            self.process_image_sequence()
            self.dirty = False

        if self.valid:
            return self.distance
        return None

    def get_scalar(self, force: bool=False) -> pint.Quantity:
        """ Get the conversion unit to multiply all new values with to conver them to the new unit

        Args:
            force (bool, optional): Force computing the new conversion scalar. Defaults to False.

        Returns:
            pint.Quantity: a quantity being of type UNIT/pixel
        """
        distance = self.get_grid_distance(force)
        if distance is None:
            return None
        return self.grid_size / UREG.Quantity(distance, 'pixel')



if __name__ == '__main__':
    import cProfile
    print('testing single grid')
    path = 'C:\\Users\\smerk\\UW\\Najafian Lab - Lab Najafian\\Foot Process Workspace\\report\\Fabry FPW-Test\\08-0050-20190328T205237Z-001\\08-0050\\08-0050 bi-2\\08_01615.tif'
    processor = GridProcessor(path, '463nm')
    print(processor.get_scalar())
    # test()