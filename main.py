import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from Preprocessing import Preprocessing
from OpticalLayoutRecognition import OpticalLayoutRecognition


def load_images(path):
    if not os.path.exists(path):
        print("Podana ścieżka nie prowadzi do folderu")
        exit()
    imgs = [img for img in os.listdir(path) if img.lower().endswith('.png')]
    if len(imgs) == 0:
        print("Podany folder nie zawiera obrazów png")
        exit()
    return imgs


def draw_histogram_lines(img, hist):
    height, width = img.shape
    draw = img.copy()
    for row in range(height):
        if hist[row] == 0:
            for column in range(width):
                draw[row, column] = 0
    return draw


def show_img_and_lines(img, img_with_segmentation, hist):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img_with_segmentation, cmap='gray')
    ax2.imshow(draw_histogram_lines(img, hist), cmap='gray')
    plt.show()


def draw_histogram_columns(img, pscs):
    height, width = img.shape
    draw = img.copy()
    for column in range(width):
        if column in pscs:
            for row in range(height):
                draw[row, column] = 0
    return draw


def show_img_and_columns(img, pscs):
    _, ax1 = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(draw_histogram_columns(img, pscs), cmap='gray')
    plt.show()


def show_img_and_hist_y(img, hist):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax2.plot(hist, range(len(hist)), color='black')
    ax2.invert_yaxis()
    ax2.set_ylim(len(hist), 0)
    ax1.imshow(img, cmap='gray')
    plt.show()


def show_img_and_hist_x(img, hist):
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
    ax2.plot(range(len(hist)), hist, color='black')
    ax2.set_xlim(0, len(hist))
    ax1.imshow(img, cmap='gray')
    plt.show()


def show_imgs(img1, img2):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img1, cmap='gray')
    ax2.imshow(img2, cmap='gray')
    plt.show()


def show_words_on_page(words_images, words_segmentation, words_coordinates, words_shapes, page):
    page_with_words = np.ones_like(page) * 255

    for word_img, seg, (x, y), (h, w) in zip(words_images, words_segmentation, words_coordinates, words_shapes):
        word_start_x = max(0, min(x, page.shape[1]))
        word_end_x = max(0, min(x + w, page.shape[1]))
        word_start_y = max(0, min(y, page.shape[0]))
        word_end_y = max(0, min(y + h, page.shape[0]))
        word_area_x = word_end_x - word_start_x
        word_area_y = word_end_y - word_start_y

        if word_img.shape[0] < word_area_y or word_img.shape[1] < word_area_x:
            word_img_resized = np.ones((word_area_y, word_area_x), dtype=np.uint8) * 255
            word_img_resized[:word_img.shape[0], :word_img.shape[1]] = word_img
        else:
            word_img_resized = word_img

        page_with_words[word_start_y:word_end_y, word_start_x:word_end_x] = word_img_resized

        seg_on_page = [x + col for col in seg]

        for col in seg_on_page:
            if 0 <= col < page.shape[1]:
                cv.line(page_with_words, (col, word_start_y), (col, word_end_y), (0, 0, 255), 1)

    plt.imshow(page_with_words)
    plt.axis('off')
    plt.show()


folder_path = "C:/Users/mariu/OneDrive/Pulpit/mg/test"
preprocessing = Preprocessing()
olr = OpticalLayoutRecognition()

images = load_images(folder_path)
for image in images:
    page = cv.imread(os.path.join(folder_path, image))

    # Przygotowanie strony pod rozpoznawanie pisma
    binary_page = preprocessing.binarize(page, 1)
    deskew_page = preprocessing.correct_skew(binary_page, 1)
    noise_page = preprocessing.remove_noises(deskew_page)
    edge_page = noise_page # preprocessing.fill_edges(noise_page)

    # Segmentacja tekstu na linie
    histogram_y = olr.create_y_histogram(edge_page)
    binary_histogram_y = olr.text_line_separation(histogram_y)
    binary_histogram_y = olr.false_line_exclusion(binary_histogram_y)
    binary_histogram_y = olr.extend_if_space_between(histogram_y, binary_histogram_y, page.shape[1], page.shape[0])

    # Segmentacja na wyrazy
    words, image_with_segmentation, image_with_words = olr.word_segmentation(edge_page, binary_histogram_y)

    # Segmentacja na znaki
    words_images, words_segmentation, words_coordinates, words_shapes = olr.character_segmentation(words, edge_page)

    # Pokazanie wyników
    show_img_and_lines(edge_page, binary_page, binary_histogram_y)
    show_imgs(edge_page, image_with_words)
    page_copy = binary_page.copy()
    show_words_on_page(words_images, words_segmentation, words_coordinates, words_shapes, page_copy)

    cv.destroyAllWindows()
