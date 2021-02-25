import cv2
import numpy as np


# from scipy.io import savemat


def normalize(im, req_mean=0.0, req_var=1.0):
    if len(im.shape) == 3:
        raise Exception("Only normalize grayscale image")
    im = im - im.mean()
    im = im / im.std()
    return req_mean + im * np.sqrt(req_var)


def pad_image(im, block_size):
    h, w = im.shape
    b_pad = block_size - h % block_size
    r_pad = block_size - w % block_size
    return cv2.copyMakeBorder(im, 0, b_pad, 0, r_pad,
                              cv2.BORDER_CONSTANT, value=0)


def ridge_segment(im, block_size, threshold):
    h, w = im.shape
    mask = np.zeros_like(im)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y: y + block_size
            roi = im[y:y + block_size, x:x + block_size]
            std = roi.std()
            mask[y:y + block_size, x:x + block_size] = std
    mask_ids = mask > threshold
    im = im - im[mask_ids].mean()
    norm_im = im / im[mask_ids].std()
    return norm_im, mask_ids


def ridge_orient(im, gradient_sigma, block_sigma, orient_smooth_sigma):
    f = gauss_kernel(gradient_sigma)
    fy, fx = np.gradient(f)
    gx = cv2.filter2D(im, -1, fx)
    gy = cv2.filter2D(im, -1, fy)
    gxx = gx * gx
    gyy = gy * gy
    gxy = gx * gy

    f = gauss_kernel(block_sigma)
    gxx = cv2.filter2D(gxx, -1, f)
    gxy = 2 * cv2.filter2D(gxy, -1, f)
    gyy = cv2.filter2D(gyy, -1, f)
    denom = np.sqrt(gxy * gxy + (gxx - gyy) ** 2)
    sin2theta = gxy / denom
    cos2theta = (gxx - gyy) / denom

    f = gauss_kernel(orient_smooth_sigma)
    cos2theta = cv2.filter2D(cos2theta, -1, f)
    sin2theta = cv2.filter2D(sin2theta, -1, f)

    return np.arctan2(sin2theta, cos2theta) / 2 + np.pi / 2  # orient along with the ridges


def cal_freq(im, orient, window_size, min_wave_len, max_wave_len):
    rows, cols = im.shape
    orient = 2 * orient.flatten()
    cos_orient = np.cos(orient).mean()
    sin_orient = np.sin(orient).mean()
    orient = np.arctan2(sin_orient, cos_orient) / 2

    # Rotate the image block so that the ridges are vertical
    angel = np.rad2deg(orient) + 90
    rotate_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angel, 1)
    rotate_im = cv2.warpAffine(im, rotate_matrix, (cols, rows), flags=cv2.INTER_NEAREST)
    # Now crop the image so that the rotated image does not contain any
    # invalid regions.  This prevents the projection down the columns
    # from being mucked up.
    crop_size = int(np.round(rows / np.sqrt(2)))
    offset = int(np.round((rows - crop_size) / 2)) - 1
    rotate_im = rotate_im[offset:offset + crop_size, offset: offset + crop_size]
    # Sum down the columns to get a projection of the grey values down the ridges
    proj = rotate_im.sum(axis=0, keepdims=True)
    # Find peaks in projected grey values by performing a greyscale
    # dilation and then finding where the dilation equals the original values.
    kernel = np.ones((window_size, window_size), np.uint8)
    dilation = cv2.dilate(proj, kernel, anchor=(-1, -1), iterations=1).round(3)
    prj_mean = proj.mean()
    proj = proj.round(3)
    max_ids = ((proj == dilation) & (proj > prj_mean)).nonzero()[1]  # test this
    num_peaks = len(max_ids)
    if num_peaks >= 2:
        wave_len = (max_ids[-1] - max_ids[0]) / (num_peaks - 1)
        if min_wave_len <= wave_len <= max_wave_len:
            return 1 / wave_len
    return 0


def ridge_frequency(im, seg_mask, orient_im, block_size,
                    window_size, min_wave_len, max_wave_len):
    h, w = im.shape
    freq = np.zeros_like(im)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block_im = im[y:y + block_size, x:x + block_size]
            block_orient = orient_im[y:y + block_size, x:x + block_size]
            block_freq = cal_freq(block_im, block_orient, window_size, min_wave_len, max_wave_len)
            freq[y:y + block_size, x:x + block_size] = block_freq
    freq = freq * seg_mask
    median_freq = np.median(freq[freq > 0])
    t = freq[freq > 0]
    freq = median_freq * seg_mask
    return freq, median_freq


def ridge_filter(im, orient, freq, kx, ky, med_freq):
    # Fixed angle increment between filter orientations in degrees.
    # This should divide evenly into 180
    rows, cols = im.shape
    angle_inc = 3
    filter_count = 180 // angle_inc
    sigma_x, sigma_y = kx / med_freq, ky / med_freq

    size = int(3 * max(sigma_x, sigma_y))
    if size % 2 != 0:
        size = size + 1
    length = 2 * size + 1
    sze = np.arange(-size, size + 1, 1)
    x, y = np.meshgrid(sze, sze)
    ref_filter1 = x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2
    ref_filter1 = np.exp(-ref_filter1 / 2)
    ref_filter2 = np.cos(2 * np.pi * med_freq * x)
    ref_filter = ref_filter1 * ref_filter2
    # Generate rotated versions of the filter.  Note orientation
    # image provides orientation *along* the ridges, hence +90
    # degrees, and imrotate requires angles +ve anticlockwise, hence
    # the minus sign.
    filters = []
    for i in range(filter_count):
        angle = - (i * angle_inc + 90)
        matrix = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
        rotated = cv2.warpAffine(ref_filter, matrix, (length, length), flags=cv2.INTER_LINEAR)
        filters.append(rotated)

    # convert orientation matrix values from radians to an index value
    # that corresponds to round(degrees/angleInc)
    # TODO: check this
    orient_index = ((filter_count / np.pi) * orient).astype(int)
    orient_mask = orient_index < 0  # in matlab code < 1
    orient_index[orient_mask] = orient_index[orient_mask] + filter_count
    orient_mask = orient_index >= filter_count
    orient_index[orient_mask] = orient_index[orient_mask] - filter_count
    # finally, find where there is valid frequency data then do the filtering
    new_im = np.zeros_like(im)
    for r in range(rows):
        for c in range(cols):
            if freq[r, c] > 0 and (size + 1 < r < rows - size - 1) and (size + 1 < c < cols - size - 1):
                new_im[r, c] = np.sum(
                    im[r - size - 1:r + size, c - size - 1:c + size] * filters[orient_index[r, c]])
    return new_im


def show(im):
    cv2.imshow('Gray image', np.uint8(im))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gauss_kernel(sigma, kernel_size=None):
    if kernel_size is None:
        kernel_size = int(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
    k1d = cv2.getGaussianKernel(kernel_size, sigma)
    return np.outer(k1d, k1d.T)


def process(im_path):
    im = cv2.imread(im_path, 0)
    equalized = cv2.equalizeHist(im)
    norm_im = normalize(equalized)

    # Step1: Ridge segment
    block_size = 24
    threshold = 0.05
    norm_im = pad_image(norm_im, block_size)
    norm_im, mask = ridge_segment(norm_im, block_size, threshold)

    gradient_sigma = 1
    block_sigma = 13
    orient_smooth_sigma = 15
    orient_im = ridge_orient(norm_im, gradient_sigma, block_sigma, orient_smooth_sigma)

    freq_im, median_freq = ridge_frequency(norm_im, mask, orient_im,
                                           block_size=36, window_size=5, min_wave_len=1, max_wave_len=25)
    filter_size = 1.9
    ridge_filter_im = ridge_filter(norm_im, orient_im, freq_im, kx=filter_size, ky=filter_size, med_freq=median_freq)

    return np.uint8(ridge_filter_im > 0)
