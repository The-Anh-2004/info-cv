""" Calculates skew angle """
import os
import imghdr
import optparse

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.transform import hough_line, hough_line_peaks


class SkewDetect:

    piby4 = np.pi / 4

    def __init__(
        self,
        input=None,
        batch_path=None,
        sigma=3.0,
        display_output=None,
        num_peaks=20,
        plot_hough=None
    ):

        self.sigma = sigma
        self.input = input
        self.batch_path = batch_path
        # self.output = output
        self.display_output = display_output
        self.num_peaks = num_peaks
        self.plot_hough = plot_hough

    def write_to_file(self, wfile, data):

        for d in data:
            wfile.write(d + ': ' + str(data[d]) + '\n')
        wfile.write('\n')

    def get_max_freq_elem(self, arr):

        max_arr = []
        freqs = {}
        for i in arr:
            if i in freqs:
                freqs[i] += 1
            else:
                freqs[i] = 1

        sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
        max_freq = freqs[sorted_keys[0]]

        for k in sorted_keys:
            if freqs[k] == max_freq:
                max_arr.append(k)

        return max_arr

    def display_hough(self, h, a, d):

        plt.imshow(
            np.log(1 + h),
            extent=[np.rad2deg(a[-1]), np.rad2deg(a[0]), d[-1], d[0]],
            cmap=plt.cm.gray,
            aspect=1.0 / 90)
        plt.show()

    def compare_sum(self, value):
        if value >= 44 and value <= 46:
            return True
        else:
            return False

    def display(self, data):

        for i in data:
            print( i + ": " + str(data[i]))

    def calculate_deviation(self, angle):

        angle_in_degrees = np.abs(angle)
        deviation = np.abs(SkewDetect.piby4 - angle_in_degrees)

        return deviation

    def check_path(self, path):

        if os.path.isabs(path):
            full_path = path
        else:
            full_path = os.getcwd() + '/' + str(path)
        return full_path

    def process_single_file(self):

        # file_path = self.check_path(self.input_file)
        res = self.determine_skew(self.input)
        return res

    def determine_skew(self, img):

        if len(img.shape) == 3:
            img = img.mean(axis=2)
        
        edges = canny(img, sigma=self.sigma)
        h, a, d = hough_line(edges)
        _, ap, _ = hough_line_peaks(h, a, d, num_peaks=self.num_peaks)

        if len(ap) == 0:
            return {"Image File": img, "Message": "Bad Quality"}

        absolute_deviations = [self.calculate_deviation(k) for k in ap]
        average_deviation = np.mean(np.rad2deg(absolute_deviations))
        ap_deg = [np.rad2deg(x) for x in ap]

        bin_0_45 = []
        bin_45_90 = []
        bin_0_45n = []
        bin_45_90n = []

        for ang in ap_deg:

            deviation_sum = int(90 - ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90.append(ang)
                continue

            deviation_sum = int(ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45.append(ang)
                continue

            deviation_sum = int(-ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45n.append(ang)
                continue

            deviation_sum = int(90 + ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90n.append(ang)

        angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
        lmax = 0

        for j in range(len(angles)):
            l = len(angles[j])
            if l > lmax:
                lmax = l
                maxi = j

        if lmax:
            ans_arr = self.get_max_freq_elem(angles[maxi])
            ans_res = np.mean(ans_arr)

        else:
            ans_arr = self.get_max_freq_elem(ap_deg)
            ans_res = np.mean(ans_arr)

        data = {
            "Image File": img,
            "Average Deviation from pi/4": average_deviation,
            "Estimated Angle": ans_res,
            "Angle bins": angles}

        if self.display_output:
            self.display(data)

        if self.plot_hough:
            self.display_hough(h, a, d)
        return data

if __name__ == '__main__':

    parser = optparse.OptionParser()

    parser.add_option(
        '-b', '--batch',
        default=None,
        dest='batch_path',
        help='Path for batch processing')
    parser.add_option(
        '-d', '--display',
        default=None,
        dest='display_output',
        help='Display logs')
    parser.add_option(
        '-i', '--input',
        default=None,
        dest='input_file',
        help='Input file name')
    parser.add_option(
        '-n', '--num',
        default=20,
        dest='num_peaks',
        help='Number of Hough Transform peaks',
        type=int)
    parser.add_option(
        '-o', '--output',
        default=None,
        dest='output_file',
        help='Output file name')
    parser.add_option(
        '-p', '--plot',
        default=None,
        dest='plot_hough',
        help='Plot the Hough Transform')
    parser.add_option(
        '-s', '--sigma',
        default=3.0,
        dest='sigma',
        help='Sigma for Canny Edge Detection',
        type=float)
    options, args = parser.parse_args()
    skew_obj = SkewDetect(
        options.input_file,
        options.batch_path,
        #options.output_file,
        options.sigma,
        options.display_output,
        options.num_peaks,
        options.plot_hough)
    skew_obj.run()
    if options.batch_path:
        batch_path = skew_obj.check_path(options.batch_path)
        if not os.path.exists(batch_path):
            print("Batch path does not exist")
            exit(1)

        for file_name in os.listdir(batch_path):
            full_file_name = os.path.join(batch_path, file_name)
            if imghdr.what(full_file_name) is not None:
                img = io.imread(full_file_name)
                skew_obj.determine_skew(img)
    else:
        if options.input_file:
            full_input_file = skew_obj.check_path(options.input_file)
            img = io.imread(full_input_file)
            skew_obj.determine_skew(img)
        else:
            print("No input file or batch path provided")