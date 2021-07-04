# distutils: language=c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
from libc cimport math
import numpy as np
import cv2

# cython imports
from calibration.array_list cimport DArrayList
from calibration.types cimport *
cimport cython
cimport numpy as np
np.import_array()

ctypedef double angle_t
ctypedef NPDOUBLE_t angle_np_t

# constants
cdef int DISTANCE_SIZE = 200  # initial arraylist size to keep track of distances
cdef NPINT32_t INVALID_DIST = -1
cdef angle_t INVALID_MEASURE = -1000  # not a valid angle/average measure
cdef int MIN_HORIZONTAL_SAMPLES = 10  # min amount of samples to capture the horizontal lines for


cpdef angle_t get_line_angle(NPINT32_t[:, :] line):
    cdef NPINT32_t x_diff, y_diff
    x_diff = line[0][2] - line[0][0]
    y_diff = line[0][3] - line[0][1]

    if x_diff == 0:
        if y_diff > 0:
            return <angle_t> (math.pi / <double> 2.0)
        elif y_diff < 0:
            return <angle_t> (-math.pi / <double> 2.0)
        return <angle_t> INVALID_MEASURE  # not a measurement
    
    # get angle otherwise
    return <angle_t> math.atan((<double> y_diff) / (<double> x_diff))


# function to find number of elements in certain range
cpdef NPINT32_t find_frequency(angle_t[:] angles, angle_t lower, angle_t upper):
    cdef NPINT32_t count = 0
    cdef angle_t angle

    # iter through slopes
    for angle in angles:
        if angle >= lower and angle < upper:
            count += 1
    
    # return count
    return count


cpdef np.ndarray[angle_np_t, ndim=1] filter_angles(angle_t[:] angles, angle_t lower, angle_t upper):
    cdef int angle_size = angles.shape[0]
    cdef DArrayList filter_angles = DArrayList(angle_size)
    cdef angle_t angle

    # iter through slopes
    for angle in angles:
        if angle != INVALID_MEASURE and angle >= lower and angle < upper:
            filter_angles.add(angle)
    
    # return count
    return filter_angles.finalize()


cpdef np.ndarray[NPUINT_t, ndim=2, mode='c'] rotate_image(np.ndarray[NPUINT_t, ndim=2, mode='c'] image, double degrees):
    cdef int width, height, center_width, center_height
    width = image.shape[1]
    height = image.shape[0]
    center_width = <int> math.floor(width / 2)
    center_height = <int> math.floor(height / 2)

    # get rotation matrix
    cdef np.ndarray M = cv2.getRotationMatrix2D((center_width, center_height), degrees, 1.0) 
    return cv2.warpAffine(image, M, (width, height))


cpdef np.ndarray[NPUINT_t, ndim=2, mode='c'] adjust_contrast(np.ndarray[NPUINT_t, ndim=2, mode='c'] img, double alpha, int beta, bool_t mean_shift=0):
    cdef int row, col, rows, cols
    rows = img.shape[0]
    cols = img.shape[1]
    cdef int value, mean
    cdef np.ndarray im_cp = np.zeros((rows, cols), dtype=np.uint8)
    cdef NPUINT_t[:, :] image_view = img
    cdef NPUINT_t[:, :] new_image = im_cp

    # iter through image and adjust contrast of the image (copy and pasted for optimized mean shift so we don't call the if a ton of times)
    if mean_shift == 1:
        mean = <int> np.mean(img)
        beta += mean  # offset beta to include the mean
        for row in range(rows):
            for col in range(cols):
                value = (<int> ((<double> (image_view[row, col] - mean)) * alpha)) + beta
                if value > 255:
                    value = 255
                elif value < 0:
                    value = 0
                new_image[row, col] = <uint8_t> value
    else:
        for row in range(rows):
            for col in range(cols):
                value = (<int> ((<double> image_view[row, col]) * alpha)) + beta
                if value > 255:
                    value = 255
                elif value < 0:
                    value = 0
                new_image[row, col] = <uint8_t> value

    return im_cp

cpdef np.ndarray[angle_np_t, ndim=1] hough_line_angles(NPINT32_t[:, :, :] lines):
    cdef int nlines = lines.shape[0]
    cdef int cur = 0
    cdef angle_t angle
    cdef np.ndarray angles = np.zeros((nlines,), np.double)

    # convert them
    for cur in range(nlines):
        angle = get_line_angle(lines[cur, :, :])
        angles[cur] = angle
    
    return angles


# function to find horizontal distance between lines in the image
cpdef np.ndarray[NPDOUBLE_t, ndim=1, mode='c'] find_horz_image_distance(NPUINT_t[:, ::1] img, double padding, int measures, int min_distance, int max_distance):
    # make sure padding is within bounds
    if padding > 1.0 or padding < 0.0:
        return np.array([], dtype=np.double)
    
    cdef int col, row, sub_col
    cdef int rows = img.shape[0]
    cdef int cols = img.shape[1]
    cdef int row_start = <int> math.floor(rows * padding)
    cdef int row_end = <int> math.floor(rows * (1.0 - padding))
    cdef int[:] samples = np.random.randint(row_start, row_end, measures)

    # numbers to keep track for averages
    cdef DArrayList distances = DArrayList(DISTANCE_SIZE)

    # temp trackers
    cdef int distance
    cdef uint8_t make_measure = 0
    cdef uint8_t in_white = 0

    # iterate through random rows
    for row in samples:
        distance = 0  # reset initial distance

        # keep iterating until we hit a drawn white line
        col = 0
        while col < cols and img[row, col] != 255:
            col += 1
        
        # we're now in white
        if img[row, col] == 255:
            in_white = 1
        
        # if we aren't at the end of the image continue
        for sub_col in range(col, cols - 1):
            if img[row, sub_col] == 255:
                if in_white == 1:  # so we're still in white let's just add to count (we expect line thickness to stay the same)
                    distance += 1
                else:  # we're in measurement and not stuck on white
                    if distance >= min_distance and distance < max_distance:
                        distances.add(<NPDOUBLE_t> distance)  # add new measurement
                    
                    # reset the counters (to also recount the lines)
                    distance = 0
                    in_white = 1
            else:  # this means we're not white
                distance += 1
                in_white = 0

    # finalize the array list
    cdef np.ndarray dist_arr = distances.finalize()

    # return the distances measured
    return dist_arr


cpdef tuple scan_angle_frequency_range(angle_t[:] angles, angle_t one, angle_t two, angle_t resolution, angle_t window, angle_t tolerance):
    """ Scans the sets of angles to figure out which ones have the most probably (with the 90 degree offsets) chance of having the most lines
    Arguments:
        angles - all of the angles listed
        one - first best angle
        two - second best angle
        resolution - how many radian increments to increase the angle
        window - the + or - range from the best (so one +- window and two +- window)
        tolerance - angle tolerance within each window scan so (one +- window) +- tolerance is the angle range (used to capture similar angles)
    """
    cdef angle_t new_window, best_one = one, best_two = two
    cdef angle_t new_one = one
    cdef angle_t new_two = two
    cdef int i = 0, freq_one = 0, freq_two = 0, best_freq_one = 0, best_freq_two = 0
    cdef int steps = <int> ((2.0 * window) / resolution)
    cdef int hsteps = <int> math.floor((<double> steps) / 2.0)

    # scan the full window
    for i in range(steps):
        new_window = ((i - hsteps) * resolution)
        new_one = one + window
        new_two = two + window

        # get the frequencies at these windows with the new tolerances
        freq_one = find_frequency(angles, new_one - tolerance, new_one + tolerance)
        freq_two = find_frequency(angles, new_two - tolerance, new_two + tolerance)

        # capture differences
        if freq_one > best_freq_one:
            best_one = new_one
            best_freq_one = freq_one

        if freq_two > best_freq_two:
            best_two = new_two
            best_freq_two = freq_two

    return (best_one, best_two)


cpdef angle_t draw_lines(np.ndarray[NPUINT_t, ndim=2, mode='c'] img, NPINT32_t[:, :, :] lines, angle_t mean, angle_t tolerance, int line_width, tuple color):
    cdef NPINT32_t[:, :] line
    cdef int cur, count = lines.shape[0], angle_count = 0
    cdef angle_t angle, total_angle = 0
    cdef int x1, y1, x2, y2

    # iterate through each line
    for cur in range(count):
        line = lines[cur, :, :]
        angle = get_line_angle(line)
        if angle != INVALID_MEASURE and angle > (mean - tolerance) and angle < (mean + tolerance):
            total_angle += angle
            angle_count += 1
        
            # draw the line
            x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2], line[0][3]
            cv2.line(img, (x1, y1), (x2, y2), color, line_width)

    if angle_count == 0:
        return INVALID_MEASURE
    
    return total_angle / (<angle_t> angle_count)