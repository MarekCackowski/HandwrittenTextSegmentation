import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
from scipy.signal import find_peaks


class OpticalLayoutRecognition:
    def __init__(self):
        self.separation_threshold = 0.8
        self.exclusion_threshold = 0.6

    # Segmentacja na wiersze

    @staticmethod
    def create_y_histogram(image):
        image = cv.bitwise_not(image)
        return np.sum(image, axis=1)

    def text_line_separation(self, histogram):
        threshold = np.mean(histogram) * self.separation_threshold
        return (histogram > threshold).astype(int)

    @staticmethod
    def add_begin_or_end(y_initial, y_final, histogram_height):
        if len(y_initial) < len(y_final):
            y_initial = np.insert(y_initial, 0, 0)
        else:
            y_final = np.append(y_final, histogram_height - 1)
        return y_initial, y_final

    @staticmethod
    def find_peaks_in_y_histogram(histogram):
        peaks, _ = find_peaks(histogram)
        return peaks

    def false_line_exclusion(self, histogram):
        y_initial = np.where(np.diff(histogram) == 1)[0]
        y_final = np.where(np.diff(histogram) == -1)[0]
        if len(y_initial) != len(y_final):
            y_initial, y_final = self.add_begin_or_end(y_initial, y_final, len(histogram))
        number_of_regions = len(self.find_peaks_in_y_histogram(histogram))
        avg_height = sum((y_final - y_initial) / number_of_regions)
        threshold = avg_height * self.exclusion_threshold
        result_histogram = histogram.copy()
        for initial, final in zip(y_initial, y_final):
            if np.abs(final - initial) < threshold:
                result_histogram[initial:final + 1] = 0
        return result_histogram

    @staticmethod
    def extend(histogram, binary_histogram, start_index, end_index, y_initial, y_final,
               mean_white_pixels_in_blank_line, threshold, i, first_center_last):
        while start_index <= end_index:
            if histogram[start_index] < mean_white_pixels_in_blank_line + threshold:
                if first_center_last > 1:
                    binary_histogram[y_final[i]:start_index] = binary_histogram[y_final[i] - 1]
                break
            start_index += 1
        while start_index <= end_index:
            if histogram[start_index] >= mean_white_pixels_in_blank_line - threshold:
                if first_center_last < 3:
                    binary_histogram[start_index:y_initial[i + 1] + 1] = binary_histogram[y_initial[i + 1] + 1]
                break
            start_index += 1
        return binary_histogram

    def extend_if_space_between(self, histogram, binary_histogram, image_width, image_height):
        threshold, minimum = 20, 0.9 * image_width
        y_initial = np.where(np.diff(binary_histogram) == 1)[0]
        y_final = np.where(np.diff(binary_histogram) == -1)[0]
        mean_white_pixels_in_blank_line = np.mean(histogram[y_final[-1] + 1:])
        if mean_white_pixels_in_blank_line < minimum:
            return binary_histogram
        for i in range(len(y_initial) - 1):
            start_index = y_final[i]
            end_index = y_initial[i + 1]
            binary_histogram = self.extend(histogram, binary_histogram, start_index, end_index, y_initial,
                                           y_final, mean_white_pixels_in_blank_line, threshold, i, 2)
        start_index, end_index = 0, y_initial[0]
        binary_histogram = self.extend(histogram, binary_histogram, start_index, end_index, y_initial, y_final,
                                       mean_white_pixels_in_blank_line, threshold, -1, 1)
        start_index, end_index = y_final[-1], image_height - 1
        binary_histogram = self.extend(histogram, binary_histogram, start_index, end_index, y_initial, y_final,
                                       mean_white_pixels_in_blank_line, threshold, -1, 3)
        return binary_histogram

    def lines_segmentation(self, preprocessed_page, page_shape):
        histogram_y = self.create_y_histogram(preprocessed_page)
        binary_histogram_y = self.text_line_separation(histogram_y)
        binary_histogram_y = self.false_line_exclusion(binary_histogram_y)
        binary_histogram_y = self.extend_if_space_between(histogram_y, binary_histogram_y, page_shape[1], page_shape[0])
        return binary_histogram_y

    # Segmentacja na wyrazy

    @staticmethod
    def delete_internal_segments(segments):
        not_internal_segments = []
        for i, segment in enumerate(segments):
            x1, y1, w1, h1 = segment
            is_internal = False
            for _, other_segment in enumerate(segments):
                x2, y2, w2, h2 = other_segment
                if x1 > x2 and x1 + w1 < x2 + w2 and y1 > y2 and y1 + h1 < y2 + h2:
                    is_internal = True
                    break
            if not is_internal:
                not_internal_segments.append([x1, y1, w1, h1])
        return not_internal_segments

    def prepare_segments(self, image):
        contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        segments = []
        for i, contour in enumerate(contours):
            if i != 0:  # pomijamy pierwszy kontur, który jest całą stroną
                x, y, w, h = cv.boundingRect(contour)
                segments.append([x, y, w, h])
        segments = self.delete_internal_segments(segments)
        return segments, contours

    @staticmethod
    def sort_segments(segments, binary_histogram):
        segments_in_lines = []
        y_initial = np.where(np.diff(binary_histogram) == 1)[0]
        y_final = np.where(np.diff(binary_histogram) == -1)[0]
        for i in range(len(y_initial)):
            segments_in_line = []
            start_index = y_initial[i]
            end_index = y_final[i]
            for j, segment in enumerate(segments):
                x, y, w, h = segment
                if start_index < y < end_index or start_index < y + h < end_index or (y <= start_index and y + h >= end_index):
                    segments_in_line.append(segment)
            segments_in_line = sorted(segments_in_line, key=lambda s: s[0])
            segments_in_lines.append(segments_in_line)
        return segments_in_lines

    @staticmethod
    def min(elements):
        return min(np.array(elements))

    @staticmethod
    def max(elements):
        return max(np.array(elements))

    def delete_duplicates(self, segments_in_lines, binary_histogram):
        y_initial = np.where(np.diff(binary_histogram) == 1)[0]
        y_final = np.where(np.diff(binary_histogram) == -1)[0]
        for i in range(len(y_initial) - 1):
            current_start_index = y_initial[i]
            current_end_index = y_final[i]
            current_line = segments_in_lines[i]
            next_start_index = y_initial[i + 1]
            next_end_index = y_final[i + 1]
            next_line = segments_in_lines[i + 1]
            for first, segment1 in enumerate(current_line):
                for second, segment2 in enumerate(next_line):
                    if segment1 == segment2:
                        current_pixels = self.min([current_end_index, segment1[1] + segment1[3]]) - self.max([current_start_index, segment1[1]])
                        next_pixels = self.min([next_end_index, segment2[1] + segment2[3]]) - self.max([next_start_index, segment2[1]])
                        if current_pixels > next_pixels:
                            del segments_in_lines[i + 1][second]
                        else:
                            del segments_in_lines[i][first]
            segments_in_lines[i] = sorted(segments_in_lines[i], key=lambda s: s[0])
        return segments_in_lines

    @staticmethod
    def calculate_average_spacing(segments):
        average_spacing = 20
        if len(segments) > 1:
            spacings = [segments[i + 1][0] - (segments[i][0] + segments[i][2]) for i in range(len(segments) - 1) if segments[i + 1][0] - (segments[i][0] + segments[i][2]) > 0]
            if spacings:
                average_spacing = (sum(spacings) / len(spacings)) * 1.1
        average_spacing = average_spacing // 1
        return average_spacing

    def group_in_words(self, segments_in_lines):
        words = []
        for segments_in_line in segments_in_lines:
            word = []
            threshold = self.calculate_average_spacing(segments_in_line)
            for i in range(len(segments_in_line) - 1):
                current_segment = segments_in_line[i]
                x_max = max(max(segment[0] + segment[2] for segment in word) if word != [] else 0, current_segment[0] + current_segment[2])
                next_segment = segments_in_line[i + 1]
                if x_max + threshold > next_segment[0]:
                    word.append(current_segment)
                    if i == len(segments_in_line) - 2:
                        word.append(next_segment)
                        words.append(word)
                else:
                    word.append(current_segment)
                    if i == len(segments_in_line) - 2:
                        word.append(next_segment)
                    words.append(word)
                    word = []
        return words

    @staticmethod
    def make_image_with_segments(image, segments_in_lines):
        image_with_segments = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        # colors = [(100, 0, 100), (0, 255, 0), (0, 0, 255)]
        color = 0
        for segments_in_line in segments_in_lines:
            for segment in segments_in_line:
                x, y, w, h = segment
                image_with_segments = cv.rectangle(image_with_segments, (x, y), (x + w, y + h), (255, 0, 0), 1)
            color = color + 1 if color != 2 else 0
        return image_with_segments

    @staticmethod
    def make_image_with_words(image, words):
        image_with_words = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        for word in words:
            x_min = min(segment[0] for segment in word)
            y_min = min(segment[1] for segment in word)
            x_max = max(segment[0] + segment[2] for segment in word)
            y_max = max(segment[1] + segment[3] for segment in word)
            image_with_words = cv.rectangle(image_with_words, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
        return image_with_words

    def word_segmentation(self, image, binary_histogram):
        segments, contours = self.prepare_segments(image)
        segments_in_lines = self.sort_segments(segments, binary_histogram)
        segments_in_lines = self.delete_duplicates(segments_in_lines, binary_histogram)
        words = self.group_in_words(segments_in_lines)
        image_with_segmentation = self.make_image_with_segments(image, segments_in_lines)
        image_with_words = self.make_image_with_words(image, words)
        return words, image_with_segmentation, image_with_words

    # Segmentacja na litery

    @staticmethod
    def get_word_image(word, image):
        x_min = min(segment[0] for segment in word)
        y_min = min(segment[1] for segment in word)
        x_max = max(segment[0] + segment[2] for segment in word)
        y_max = max(segment[1] + segment[3] for segment in word)
        word_image = image[y_min:y_max, x_min:x_max]
        mask = np.zeros_like(word_image, dtype=np.uint8)
        for segment in word:
            segment_x_min, segment_y_min, segment_width, segment_height = segment
            mask[segment_y_min - y_min:segment_y_min - y_min + segment_height, segment_x_min - x_min:segment_x_min - x_min + segment_width] = 255
        word_image[~mask.astype(bool)] = 255
        return word_image, [x_min, y_min]

    @staticmethod
    def mark_border_contour(segment_image, is_top, height):
        function = np.min if is_top else np.max
        highest_black_pixels = []
        for column in range(segment_image.shape[1]):
            black_pixels = np.where(segment_image[:, column] == 0)[0]
            if black_pixels.size > 0:
                index_of_highest_black_pixel = function(black_pixels)
                highest_black_pixels.append(index_of_highest_black_pixel)
            else:
                highest_black_pixels.append(0 if is_top else height - 1)
        return highest_black_pixels

    def top_and_bottom_contour(self, segment_image):
        height = segment_image.shape[0]
        top_contour = self.mark_border_contour(segment_image, True, height)
        bottom_contour = self.mark_border_contour(segment_image, False, height)
        return top_contour, bottom_contour

    @staticmethod
    def find_definite_peaks(bottom_peaks, height):
        definite_peaks = []
        for bottom_peak in bottom_peaks:
            if bottom_peak == height - 1:
                definite_peaks.append(bottom_peak)
        definite_peaks = np.array(definite_peaks)
        return definite_peaks

    def find_peaks_in_contour(self, contour, is_top, height):
        contour = np.array(contour)
        contour *= is_top
        contour += 0 if is_top == 1 else height
        peaks, _ = find_peaks(contour, 7)
        definite_peaks = []
        if is_top == -1:
            definite_peaks, _ = find_peaks(contour, 20)
            definite_peaks = np.concatenate((definite_peaks, self.find_definite_peaks(contour, height)))
            definite_peaks.sort()
        return peaks, definite_peaks

    @staticmethod
    def skeletonize(segment_image):
        pcv.params.line_thickness = 3
        skeleton = pcv.morphology.skeletonize(mask=cv.bitwise_not(segment_image))
        skeleton = cv.bitwise_not(skeleton)
        return skeleton

    @staticmethod
    def calculate_top_line(top_contour, height):
        top_contour = [max_row for max_row in top_contour if max_row != 0]
        for i in range(len(top_contour)):
            top_contour[i] = top_contour[i] if top_contour[i] < height // 2 else height // 2
        top_median = sum(top_contour) / len(top_contour) // 1
        top_median = max(top_median - 5, 10)
        return top_median

    def prepare_letter_parts(self, word_image, peaks, begin, end, is_whole):
        segment = word_image[:, peaks[begin]: peaks[end] if is_whole else peaks[begin] + end]
        height, width = segment.shape[:2]
        bordered_image = np.ones((height + 2, width + 2), dtype=np.uint8) * 255
        bordered_image[1:height + 1, 1:width + 1] = segment
        letter_parts, _ = self.prepare_segments(bordered_image)
        return letter_parts

    def merge_capital_letter(self, bottom_peaks, segment_image):
        skeleton = self.skeletonize(segment_image)
        new_bottom_peaks = np.concatenate((bottom_peaks, [segment_image.shape[1] - 1]))
        top_contour, bottom_contour = self.top_and_bottom_contour(skeleton)
        top_contour = np.array(top_contour)
        top_line = int(self.calculate_top_line(top_contour, segment_image.shape[0]))
        # print('top_line', top_line)
        is_capital = False
        to_delete = []
        if segment_image.shape[0] > 50:
            for i in range(min(25, segment_image.shape[1] - 1)):
                if (top_contour[i] < top_line) and top_contour[i] != 0:
                    is_capital = True
        if is_capital:
            for i in range(1, len(new_bottom_peaks) - 1):
                if new_bottom_peaks[i] < 40 and len(np.where(skeleton[:top_line, self.max([0, new_bottom_peaks[i] - 5]):new_bottom_peaks[i + 1]] == 0)[0]) >= 10:
                    # print(new_bottom_peaks[i])
                    letter_parts = self.prepare_letter_parts(segment_image, new_bottom_peaks, 0, i + 1, True)
                    if np.all(top_contour[self.max([0, new_bottom_peaks[i] - 3]):new_bottom_peaks[i + 1]] != 0) or len(letter_parts) < 2:
                        to_delete.append(i)
        to_delete.sort(reverse=True)
        for i in to_delete:
            new_bottom_peaks = np.delete(new_bottom_peaks, i)
        return new_bottom_peaks

    @staticmethod
    def create_x_histogram(segment_image):
        reversed_image = cv.bitwise_not(segment_image)
        histogram = np.sum(reversed_image // 255, axis=0)
        return histogram

    @staticmethod
    def calculate_legs(bottom_contour, bottom_value, first_top_peak, middle_top_peak, last_top_peak, segment_width):
        begin = max(first_top_peak - 5, 0)
        mask = bottom_contour[begin: first_top_peak + 5] != bottom_value
        left_leg = np.max(bottom_contour[begin: first_top_peak + 5][mask]) if np.any(mask) else 0
        mask = bottom_contour[middle_top_peak - 5: middle_top_peak + 5] != bottom_value
        middle_leg = np.max(bottom_contour[middle_top_peak - 5: middle_top_peak + 5][mask]) if np.any(mask) else 0
        end = min(last_top_peak + 5, segment_width - 1)
        mask = bottom_contour[last_top_peak - 5: end] != bottom_value
        right_leg = np.max(bottom_contour[last_top_peak - 5: end][mask]) if np.any(mask) else 0
        return left_leg, middle_leg, right_leg

    @staticmethod
    def calculate_diff_counts(skeleton, bottom_contour, top_contour, first_top_peak, middle_top_peak, last_top_peak, top_value):
        diff_count_left, diff_count_right = 0, 0
        for i in range(max(first_top_peak - 5, 0), middle_top_peak):
            if top_contour[i] != top_value:
                white_pixels_between = len(np.where(skeleton[top_contour[i]:bottom_contour[i], i] == 255)[0])
                if white_pixels_between > 5:
                    diff_count_left += 1
        for i in range(middle_top_peak, min(last_top_peak + 5, skeleton.shape[1] - 1)):
            if top_contour[i] != top_value:
                white_pixels_between = len(np.where(skeleton[top_contour[i]:bottom_contour[i], i] == 255)[0])
                if white_pixels_between > 5:
                    diff_count_right += 1
        return diff_count_left, diff_count_right

    @staticmethod
    def assign_new_peaks(new_bottom_peaks, first_top_peak, first_bottom_peak, last_bottom_peak):
        new_bottom_peaks = np.delete(new_bottom_peaks, last_bottom_peak)
        new_bottom_peaks = np.delete(new_bottom_peaks, first_bottom_peak)
        if all(first_top_peak - i not in new_bottom_peaks for i in range(0, 10)):
            new_bottom_peaks = np.concatenate((new_bottom_peaks, [max(0, first_top_peak - 5)]))
        new_bottom_peaks = np.sort(new_bottom_peaks)
        return new_bottom_peaks

    def repair_over_segmentation_of_w_or_m(self, top_peaks, bottom_peaks, segment_image, is_top_first):
        segment_height, segment_width = segment_image.shape[0:2]
        to_delete_bottom, middle_top_peaks_to_delete = [], []
        skeleton = self.skeletonize(segment_image)
        new_bottom_peaks = bottom_peaks.copy()
        new_top_peaks = top_peaks.copy()
        top_value, bottom_value = 0, segment_height - 1
        if is_top_first:
            top_contour, bottom_contour = self.top_and_bottom_contour(skeleton)
        else:
            skeleton = skeleton[::-1]
            top_contour, bottom_contour = self.top_and_bottom_contour(skeleton)
        top_contour, bottom_contour = np.array(top_contour), np.array(bottom_contour)
        new_top_peaks = np.concatenate((new_top_peaks, [segment_width - 1]))
        new_bottom_peaks = np.concatenate((new_bottom_peaks, [segment_width - 1]))
        i = 0
        while i < len(new_top_peaks) - 2:
            first_top_peak, middle_top_peak, last_top_peak = new_top_peaks[i:i + 3]
            while first_top_peak < middle_top_peak and top_contour[first_top_peak] == top_value:
                first_top_peak += 1
            while last_top_peak > middle_top_peak and top_contour[last_top_peak] == top_value:
                last_top_peak -= 1
            # print('top_peaks', first_top_peak, middle_top_peak, last_top_peak)
            if 40 < last_top_peak - first_top_peak < 100:
                first_bottom_peaks = np.where((new_bottom_peaks > first_top_peak) & (new_bottom_peaks < middle_top_peak))[0]
                last_bottom_peaks = np.where((new_bottom_peaks > middle_top_peak) & (new_bottom_peaks < last_top_peak))[0]
                # print(first_bottom_peaks, last_bottom_peaks)
                if len(first_bottom_peaks) == 1 and len(last_bottom_peaks) == 1 and top_contour[middle_top_peak] != top_value:
                    first_bottom = bottom_contour[new_bottom_peaks[first_bottom_peaks[0]]]
                    last_bottom = bottom_contour[new_bottom_peaks[last_bottom_peaks[0]]]
                    # print('bottom_peaks', first_bottom, last_bottom)
                    if first_bottom != bottom_value and last_bottom != bottom_value:
                        left_leg, middle_leg, right_leg = self.calculate_legs(bottom_contour, bottom_value, first_top_peak, middle_top_peak, last_top_peak, segment_width)
                        # print('legs', left_leg, middle_leg, right_leg)
                        if first_bottom < middle_leg and last_bottom < middle_leg:
                            if first_bottom + 7 < left_leg and first_bottom + 3 < right_leg:
                                # print('last_top')
                                if last_bottom + 7 < right_leg and last_bottom + 3 < left_leg:
                                    diff_count_left, diff_count_right = self.calculate_diff_counts(skeleton, bottom_contour, top_contour, first_top_peak, middle_top_peak, last_top_peak, top_value)
                                    if last_top_peak > middle_top_peak > first_top_peak and top_value < first_bottom - 1 and top_value < last_bottom - 1:
                                        first_histogram = self.create_y_histogram(skeleton[top_value:first_bottom - 1, first_top_peak:middle_top_peak])
                                        second_histogram = self.create_y_histogram(skeleton[top_value:last_bottom - 1, middle_top_peak:last_top_peak])
                                        # print(first_histogram, second_histogram)
                                        if diff_count_right <= 5 and diff_count_left <= 5:
                                            if sum(first_histogram) // 255 < 5 and sum(second_histogram) // 255 < 5:
                                                # print(left_leg, right_leg, right_leg, first_bottom, last_bottom, diff_count_right, diff_count_left)
                                                letter_parts = self.prepare_letter_parts(segment_image, new_top_peaks, i, i + 2, True)
                                                # print(letter_parts)
                                                if len(letter_parts) < 2:
                                                    middle_top_peaks_to_delete.append(middle_top_peak)
                                                    to_delete_bottom.append([first_top_peak, first_bottom_peaks[0], last_bottom_peaks[0]])
                                                    i += 2
            i += 1
        for row in to_delete_bottom:
            new_bottom_peaks = self.assign_new_peaks(new_bottom_peaks, *row)
            # print(row)
        for row in middle_top_peaks_to_delete:
            new_top_peaks = new_top_peaks[new_top_peaks != row]
        return new_top_peaks[:-1], new_bottom_peaks[:-1]

    @staticmethod
    def count_white_pixels(skeleton, column, bottom_contour, top_contour):
        bottom_black_pixel, top_black_pixel = bottom_contour[column], top_contour[column]
        white_pixels_between = 0
        if top_black_pixel != 0:
            for row in range(top_black_pixel, bottom_black_pixel):
                if skeleton[row, column] == 255:
                    white_pixels_between += 1
        return white_pixels_between

    def adjust_letter_e_or_c(self, word_image, peaks):
        adjusted_peaks = peaks.copy()
        new_word_image = word_image.copy()
        skeleton = self.skeletonize(new_word_image)
        top_contour, bottom_contour = self.top_and_bottom_contour(skeleton)
        top_contour = np.array(top_contour)
        for i in range(len(adjusted_peaks)):
            column, begin = adjusted_peaks[i], adjusted_peaks[i]
            # print(column, self.count_white_pixels(skeleton, column, bottom_contour, top_contour))
            if column > 0 and 3 <= self.count_white_pixels(skeleton, column, bottom_contour, top_contour) <= 30:
                column -= 1
                while column > 0 and self.count_white_pixels(skeleton, column, bottom_contour, top_contour) != 0:
                    column -= 1
                # print(column)
                if begin - column < 10 and self.count_white_pixels(skeleton, column - 1, bottom_contour, top_contour) <= 30:
                    adjusted_peaks[i] = max(0, column - 3)
        return adjusted_peaks

    def repair_over_segmentation_of_n(self, bottom_peaks, top_peaks, width, segment_image):
        skeleton = self.skeletonize(segment_image)
        histogram = self.create_x_histogram(skeleton)
        top_contour, bottom_contour = self.top_and_bottom_contour(skeleton)
        top_contour, bottom_contour = np.array(top_contour), np.array(bottom_contour)
        top_peaks = np.concatenate((top_peaks, [width - 1]))
        for bottom_peak in bottom_peaks:
            left_peak = max((peak for peak in top_peaks if peak < bottom_peak), default=None)
            right_peak = min((peak for peak in top_peaks if peak > bottom_peak), default=None)
            if left_peak and right_peak:
                if np.all(histogram[left_peak + 1:right_peak - 1] > 0) and (bottom_peak - left_peak < 25 or top_contour[max(0, left_peak - 5)] == 0) and (right_peak - bottom_peak < 30 or top_contour[min(width - 1, right_peak + 5)] == 0) and right_peak - left_peak > 25:
                    left_leg, _, right_leg = self.calculate_legs(bottom_contour, segment_image.shape[0] - 1, left_peak, left_peak, right_peak, segment_image.shape[1])
                    # print('legs', bottom_peak, left_leg, right_leg, bottom_contour[bottom_peak])
                    # print(bottom_peak, left_peak, histogram[left_peak - 10: left_peak + 2])
                    current = left_peak
                    while current > 0 and top_contour[current] <= top_contour[current - 1] and top_contour[current] != 0:
                        current -= 1
                    while current > 0 and top_contour[current] >= top_contour[current - 1] and top_contour[current] != 0:
                        current -= 1
                    # print(bottom_peak, top_contour[current], top_contour[left_peak] + 5, top_contour[bottom_peak] + 5)
                    # print(bottom_peak, bottom_contour[bottom_peak] + 50, left_leg, bottom_contour[bottom_peak] + 15, right_leg)
                    if right_leg > bottom_contour[bottom_peak] + 15 and (left_leg > bottom_contour[bottom_peak] + 15 or np.any(histogram[max(0, left_peak - 10): left_peak + 2] > 7)):
                        if np.any(histogram[max(0, left_peak - 10): left_peak + 2] > 4) and (top_contour[current] > top_contour[bottom_peak] + 5 or top_contour[current] == 0 or top_contour[max(0, current - 5)] == 0 or top_contour[max(0, current - 5)] < top_contour[bottom_peak] + 5):
                            diff_count_left, diff_count_right = self.calculate_diff_counts(skeleton, bottom_contour, top_contour, left_peak, bottom_peak, right_peak, 0)
                            if self.count_white_pixels(skeleton, right_peak, bottom_contour, top_contour) < 5 and diff_count_left < 5 and diff_count_right < 5:
                                # print(bottom_peak, 'n')
                                bottom_peaks = bottom_peaks[bottom_peaks != bottom_peak]
                                if all(right_peak + i not in bottom_peaks for i in range(0, 10)):
                                    bottom_peaks = np.concatenate((bottom_peaks, [right_peak + 5]))
                                    bottom_peaks = np.sort(bottom_peaks)
        return bottom_peaks

    @staticmethod
    def calculate_white_pixels(skeleton, left_peak, right_peak, bottom_contour, top_contour):
        white_pixels_between = 0
        for column in range(left_peak, right_peak):
            bottom_black_pixel, top_black_pixel = bottom_contour[column], top_contour[column]
            is_white_pixel_between = False
            if top_black_pixel != 0:
                for row in range(top_black_pixel, bottom_black_pixel):
                    if skeleton[row, column] == 255:
                        is_white_pixel_between = True
            if is_white_pixel_between is True:
                white_pixels_between += 1
        return white_pixels_between

    def assign_tails_to_letter(self, word_image, peaks):
        adjusted_peaks = peaks.copy()
        new_word_image = word_image.copy()
        new_peaks = np.concatenate((peaks, [word_image.shape[1] - 1]))
        skeleton = self.skeletonize(new_word_image)
        top_contour, bottom_contour = self.top_and_bottom_contour(skeleton)
        top_contour = np.array(top_contour)
        for i in range(1, len(new_peaks) - 1):
            end = new_peaks[i + 1]
            while end > 0 and top_contour[end] == 0:
                end -= 1
            # print(new_peaks[i], end)
            is_adjusted = False
            if np.all(top_contour[self.max([new_peaks[i] - 5, 0]):new_peaks[i]] != 0) and new_peaks[i] - new_peaks[i - 1] < 40:
                if end - new_peaks[i] < 20:
                    if end + 10 > word_image.shape[1] - 1 or end != new_peaks[i + 1] or bottom_contour[new_peaks[i + 1]] - top_contour[new_peaks[i + 1]] > 30:
                        letter_parts = self.prepare_letter_parts(new_word_image, new_peaks, i, end - new_peaks[i], False)
                        # print(letter_parts)
                        if len(letter_parts) < 2:
                            adjusted_peaks[i] = end
                            is_adjusted = True
                if not is_adjusted:
                    current = self.min([new_peaks[i] + 5, word_image.shape[1] - 1])
                    # print(new_peaks[i], current, max(top_contour[self.max([new_peaks[i] - 3, 0]): new_peaks[i] + 3]))
                    while current < word_image.shape[1] - 1 and (bottom_contour[current] <= bottom_contour[current + 1] and top_contour[current] <= top_contour[current + 1] and top_contour[current] != 0) or top_contour[current] - 10 > min(top_contour[self.max([new_peaks[i] - 3, 0]): new_peaks[i] + 3]):
                        current += 1
                    white_pixels_between = self.calculate_white_pixels(skeleton, self.max([new_peaks[i], 0]), current, bottom_contour, top_contour)
                    if (current + 1 > word_image.shape[1] - 1 or top_contour[current + 1] == 0) and white_pixels_between < 5:
                        letter_parts = self.prepare_letter_parts(new_word_image, new_peaks, i, current - new_peaks[i], False)
                        # print(new_peaks[i], current, letter_parts)
                        if len(letter_parts) < 2:
                            adjusted_peaks[i] = current
        return adjusted_peaks

    @staticmethod
    def calculate_last_line(contour, new_peaks, height, i, is_lowest):
        last_line = 0 if is_lowest else (height - 1)
        if len(new_peaks) > 1:
            for value in contour[new_peaks[i]:new_peaks[i + 1] - 3]:
                temp = (height - 1) if is_lowest else 0
                if value != temp:
                    last_line = max(last_line, value) if is_lowest else min(last_line, value)
        return last_line

    def assign_long_tails(self, word_image, peaks):
        adjusted_peaks = peaks.copy()
        new_word_image = word_image.copy()
        new_peaks = np.concatenate((peaks, [word_image.shape[1]]))
        skeleton_image = self.skeletonize(new_word_image)
        top_contour, bottom_contour = self.top_and_bottom_contour(skeleton_image)
        to_delete = []
        lowest_line = self.calculate_last_line(bottom_contour, new_peaks, new_word_image.shape[0], 0, True)
        for i in range(1, len(new_peaks) - 1):
            # print(lowest_line)
            if self.min([new_peaks[i] + 7, new_peaks[i + 1]]) - self.max([0, new_peaks[i] - 2]) > 0:
                histogram = self.create_y_histogram(skeleton_image[lowest_line:, self.max([0, new_peaks[i] - 3]):new_peaks[i + 1]])
                # print(histogram)
                if top_contour[new_peaks[i] - 1] != 0:
                    if np.count_nonzero(histogram) > 20 or np.sum(new_word_image[lowest_line:, new_peaks[i]:new_peaks[i + 1]] == 0) > 0.7 * np.sum(new_word_image[:, new_peaks[i]:new_peaks[i + 1]] == 0):
                        letter_parts = self.prepare_letter_parts(new_word_image, new_peaks, i, i + 1, True)
                        if len(letter_parts) < 2:
                            # print('long tail', new_peaks[i], lowest_line)
                            to_delete.append(i)
                        else:
                            letter_parts = self.prepare_letter_parts(new_word_image, new_peaks, i, 15, False)
                            if len(letter_parts) < 2:
                                adjusted_peaks[i] = adjusted_peaks[i] + 15
            lowest_line = self.calculate_last_line(bottom_contour, new_peaks, new_word_image.shape[0], i, True)
        to_delete.sort(reverse=True)
        for i in to_delete:
            adjusted_peaks = np.delete(adjusted_peaks, i)
        return adjusted_peaks

    def adjust_peaks_for_tails(self, word_image, peaks):
        temp_peaks1 = self.assign_tails_to_letter(word_image, peaks)
        adjusted_peaks = self.assign_long_tails(word_image, temp_peaks1)
        return adjusted_peaks

    def clean_ligature_segments(self, word_image, peaks):
        new_word_image = word_image.copy()
        skeleton_image = self.skeletonize(new_word_image)
        top_contour, bottom_contour = self.top_and_bottom_contour(skeleton_image)
        for i in range(len(peaks) - 1):
            if peaks[i] < peaks[i + 1]:
                is_ligature = True
                prev_column = 0
                for column in range(peaks[i], peaks[i + 1]):
                    if bottom_contour[column] - top_contour[column] > 3 and top_contour[column] != 0:
                        is_ligature = False
                    if abs(prev_column - top_contour[column]) > 3 and prev_column != 0 and top_contour[column] != 0:
                        is_ligature = False
                    prev_column = top_contour[column]
                    if not is_ligature:
                        break
                if is_ligature:
                    # print(1)
                    new_word_image[:, peaks[i]:peaks[i + 1]] = 255
        return new_word_image

    @staticmethod
    def delete_and_adjust(new_word_image, skeleton, peaks, i):
        if peaks[i + 1] < new_word_image.shape[1]:
            diff = peaks[i + 1] - peaks[i]
            new_word_image = np.delete(new_word_image, range(peaks[i], peaks[i + 1]), axis=1)
            skeleton = np.delete(skeleton, range(peaks[i], peaks[i + 1]), axis=1)
            for j in range(i + 1, len(peaks)):
                peaks[j] -= diff
            new_peaks = np.delete(peaks, i + 1)
        else:
            new_peaks = peaks[peaks < new_word_image.shape[1]]
        return new_word_image, new_peaks, skeleton

    def delete_white_and_small_segments(self, word_image, skeleton, bottom_peaks, top_peaks):
        new_word_image = word_image.copy()
        new_peaks = np.concatenate((bottom_peaks, [word_image.shape[1] - 1]))
        new_top_peaks = top_peaks if top_peaks is not None else []
        i = 0
        while i < len(new_peaks) - 1:
            if new_peaks[i + 1] - new_peaks[i] > 5:
                segment = skeleton[:, new_peaks[i] + 1:new_peaks[i + 1] - 1]
                histogram = self.create_y_histogram(segment)
                # print(new_peaks[i], new_peaks[i + 1], histogram, new_peaks)
                if np.count_nonzero(segment == 0) < 15 and np.count_nonzero(histogram) < 10:
                    new_word_image, new_peaks, skeleton = self.delete_and_adjust(new_word_image, skeleton, new_peaks, i)
                else:
                    i += 1
            else:
                new_word_image, new_peaks, skeleton = self.delete_and_adjust(new_word_image, skeleton, new_peaks, i)
        return new_word_image, new_peaks[:-1], new_top_peaks

    def merge_peaks(self, peaks, word_image, definite_peaks):
        merge_peaks = []
        skeleton = self.skeletonize(word_image)
        threshold = 10
        if len(peaks) > 0:
            current_group = [0]
            for i in range(1, len(peaks)):
                if peaks[i] - current_group[-1] <= threshold and not np.any(
                  skeleton[peaks[i]:current_group[-1]]) > 2 and peaks[i] not in definite_peaks:
                    current_group.append(peaks[i])
                else:
                    merge_peaks.append(np.mean(current_group).astype(int))
                    current_group = [peaks[i]]
            merge_peaks.append(np.mean(current_group).astype(int))
        else:
            merge_peaks.append(0)
        merge_peaks = np.array(merge_peaks)
        return merge_peaks

    def character_segmentation(self, words, image):
        words_images, words_segmentation = [], []
        words_coordinates, words_shapes = [], []
        for word in words:
            word_image, word_coordinates = self.get_word_image(word, image)
            height, width = word_image.shape
            top_contour, bottom_contour = self.top_and_bottom_contour(word_image)
            top_peaks, _ = self.find_peaks_in_contour(top_contour, 1, height)
            bottom_peaks, definite_peaks = self.find_peaks_in_contour(bottom_contour, -1, height)
            # bottom_peaks = self.merge_peaks(bottom_peaks, word_image, definite_peaks)
            new_bottom_peaks = self.merge_capital_letter(bottom_peaks, word_image)
            new_top_peaks, new_bottom_peaks = self.repair_over_segmentation_of_w_or_m(top_peaks, new_bottom_peaks, word_image, True)
            new_bottom_peaks, new_top_peaks = self.repair_over_segmentation_of_w_or_m(new_bottom_peaks, new_top_peaks, word_image, False)
            new_bottom_peaks = self.adjust_letter_e_or_c(word_image, new_bottom_peaks)
            peaks_to_consider = self.repair_over_segmentation_of_n(new_bottom_peaks, new_top_peaks, width, word_image)
            adjusted_peaks = self.adjust_peaks_for_tails(word_image, peaks_to_consider)
            skeleton = self.skeletonize(word_image)
            peaks_for_img = adjusted_peaks
            new_word_image = self.clean_ligature_segments(word_image, adjusted_peaks)
            new_word_image, new_peaks_to_consider, _ = self.delete_white_and_small_segments(new_word_image, self.skeletonize(new_word_image), adjusted_peaks, None)

            words_images.append(new_word_image)
            words_segmentation.append(new_peaks_to_consider)
            words_coordinates.append(word_coordinates)
            words_shapes.append([new_word_image.shape[0], new_word_image.shape[1]])

            def show_contour(img, contour):
                temp = img.copy()
                height, width = temp.shape[0:2]
                temp[:, :] = 255
                for row in range(width):
                    for column in range(height):
                        if contour[row] == column:
                            temp[column, row] = 0
                return temp

            def draw_contour_columns(img, peaks):
                height, width = img.shape
                draw = img.copy()
                for peak in peaks:
                    for row in range(height):
                        draw[row, peak] = 0
                return draw

            def show_img_and_contour(skeleton, img, new_image, peaks1, peaks2, peaks3):
                _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 6))
                ax1.imshow(draw_contour_columns(skeleton, peaks1), cmap='gray')
                ax2.imshow(draw_contour_columns(img, peaks2), cmap='gray')
                ax3.imshow(new_image, cmap='gray')
                plt.show()

            # _, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 6))
            # ax1.imshow(word_image, cmap='gray')
            # ax2.imshow(skeleton, cmap='gray')
            # plt.show()

            # _, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 6))
            # ax1.imshow(draw_contour_columns(word_image, bottom_peaks), cmap='gray')
            # ax2.imshow(draw_contour_columns(show_contour(word_image, bottom_contour), bottom_peaks), cmap='gray')
            # plt.show()

            show_img_and_contour(skeleton, word_image, word_image, peaks_to_consider, peaks_for_img, peaks_for_img)
            # print("\n" * 100)

        return words_images, words_segmentation, words_coordinates, words_shapes
