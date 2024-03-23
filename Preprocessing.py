import numpy as np
import cv2 as cv


class Preprocessing:
    def __init__(self):
        self.binarize_method = [self.gaussian_adaptive_thresholding, self.otsu_thresholding]
        self.correct_skew_method = [self.hough_transform_method, self.fft_method]
        self.ksize = 3
        self.kernel_size = 2
        self.iterations_dilate = 1
        self.iterations_erode = 1

    # Binaryzacja

    @staticmethod
    def gaussian_adaptive_thresholding(image):
        block_size, c = 11, 2
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, c)
        return image

    @staticmethod
    def otsu_thresholding(image):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray_histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 255])
        gray_histogram = gray_histogram.astype(float) / gray_histogram.sum()
        sigmas = []
        for k in range(256):
            mi = []
            for i in range(256):
                mi.append(i * gray_histogram[i])
            mi_t = np.cumsum(mi)[255]
            omega_k = np.cumsum(gray_histogram)[k]
            mi_k = np.cumsum(mi)[k]
            sigma_b_squared = 0 if (omega_k * (1 - omega_k)) == 0 else (mi_t * omega_k - mi_k) ** 2 / (omega_k * (1 - omega_k))
            sigmas.append(sigma_b_squared)
        optimal_k = np.mean(np.argmax(sigmas))
        optimal_k = round(optimal_k.item())
        _, image = cv.threshold(image, optimal_k, 255, cv.THRESH_BINARY)
        return image

    def binarize(self, image, method):
        return self.binarize_method[method](image)

    # Korekcja pochylenia

    @staticmethod
    def magnitude_spectrum_in_line(angle_in_radians, magnitude_spectrum, center):
        vectorize = np.vectorize(lambda x: magnitude_spectrum[center + int(x * np.cos(angle_in_radians)),
                                                              center + int(-1 * x * np.sin(angle_in_radians))])
        return np.sum(vectorize(range(0, center)))

    def calculate_angle(self, magnitude_spectrum):
        center = magnitude_spectrum.shape[0] // 2
        angle_max, number_of_points = 25, 20
        angles_range = np.radians(np.linspace(-1 * angle_max, angle_max, int(angle_max * number_of_points * 2)))
        angles_range_copy = angles_range.copy()
        vectorize_function = np.vectorize(self.magnitude_spectrum_in_line, excluded=['magnitude_spectrum', 'center'])
        vectorization_results = vectorize_function(angles_range_copy,
                                                   magnitude_spectrum=magnitude_spectrum, center=center)
        angle = np.degrees(angles_range[np.argmax(vectorization_results)])
        return angle if -1 * angle_max != angle else 0

    def fft_method(self, image):
        optimal_dft_size = cv.getOptimalDFTSize(max(image.shape[:2]))
        optimal_dft_image = cv.copyMakeBorder(image, 0, optimal_dft_size - image.shape[0], 0,
                                              optimal_dft_size - image.shape[1], cv.BORDER_CONSTANT, value=[255.0])
        dft = np.fft.fft2(optimal_dft_image)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = np.abs(dft_shift)
        angle = self.calculate_angle(magnitude_spectrum)
        return angle

    @staticmethod
    def hough_transform_method(image):
        edge_image = cv.Canny(image=image, threshold1=100, threshold2=200)
        edge_image = cv.bitwise_not(edge_image)
        AA = 15
        R1, C1 = 0, 0
        R2, C2 = edge_image.shape
        D_max = int(np.sqrt(R2 ** 2 + C2 ** 2))
        T = [[] for _ in range(0, D_max * 10)]
        for D in range(0, D_max * 10):
            T[D] = [None if theta % 2 == 1 else 0 for theta in range(0, AA * 10)]
        for x in range(R1, R2):
            for y in range(C1, C2):
                if (edge_image[x][y]) == 0:
                    theta = 0
                    while theta < AA * 10:
                        D = round(y * np.sin(np.radians(theta / 10)) + x * np.cos(np.radians(theta / 10)), 0)
                        T[int(D)][int(theta)] += 1
                        theta += 2
        vote_max = 0
        theta1 = 0
        D = 0
        while D < D_max * 10:
            theta = 0
            while theta < AA * 10:
                if T[D][theta] >= vote_max:
                    vote_max = T[D][theta]
                    theta1 = theta
                theta += 2
            D += 1
        skewed_angle = -theta1 / 10
        return skewed_angle

    def correct_skew(self, image, method):
        angle = self.correct_skew_method[method](image)
        height, width = image.shape[:2]
        matrix = cv.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
        skew_image = cv.warpAffine(image, matrix, (width, height), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT,
                                   borderValue=[255.0])
        return skew_image

    # Usuwanie szumów

    def salt_and_pepper_noise_removal(self, image):
        image = cv.medianBlur(image, self.ksize)
        return image

    @staticmethod
    def marginal_noise_removal(image):
        # TODO
        return image

    @staticmethod
    def ruled_line_noise_removal(image):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
        return image

    def remove_noises(self, image):
        # image = self.salt_and_pepper_noise_removal(image)
        image = self.marginal_noise_removal(image)
        image = self.ruled_line_noise_removal(image)
        return image

    # Wypełnianie krawędzi

    # Najpierw poszerzamy białe obszary, a następnie
    # zmniejszamy je, usuwając piksele z ich otoczenia.
    # Powoduje to, że niewielkie przerwy między krawędziami
    # zostają zapełnione, a krawędź staje się bardziej zwarta
    def fill_edges(self, image):
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        image = cv.erode(image, kernel, iterations=self.iterations_erode)
        image = cv.dilate(image, kernel, iterations=self.iterations_dilate)
        return image

    # Skalowanie

    def rescale(self, image):
        # TODO
        return
