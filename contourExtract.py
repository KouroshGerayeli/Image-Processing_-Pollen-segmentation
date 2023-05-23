from sklearn import cluster
import cv2
import matplotlib.pyplot as plt
import skimage.filters as skf
import numpy as np
import skimage.morphology as skm
from scipy import ndimage as ndi
from skimage.segmentation import watershed, find_boundaries
from skimage.feature import peak_local_max
from skimage.morphology import reconstruction
import skimage.color as skc
import skimage.util as sku
import skimage.io as skio

plt.rcParams['image.cmap'] = 'gray'
plt.close('all')


def contourExtract(img, sig = 10, med = 20, closing_fp1 = skm.disk(15), fp1 = np.ones((60, 60)), min1 = 4000, max1 = 30000,
                   closing_fp2 = skm.disk(5), fp2 = np.ones((90, 90)), return_kmeans = False, remove_holes = 120, filename = np.random.randint(1, 50)):
    raw_img = img.copy()
    img = sku.img_as_float(skc.rgb2gray(img))
    # Median + Gaussian Filtering the image to remove some of the noise
    img = skf.median(img, footprint = np.ones((med, med)))
    img = skf.gaussian(img, sigma = sig)

    # Plotting the Noise Filtering Part
    # plt.figure()
    # plt.subplot(1, 2, 1), plt.imshow(raw_img), plt.title('Raw Image')
    # plt.subplot(1, 2, 2), plt.imshow(img), plt.title('Median (20, 20) + Gaussian (Sig = 10)')
    
    # Performing K-means Clustering
    X = img.reshape((-1,1))
    k_means = cluster.KMeans(n_clusters=2)
    k_means.fit(X)
    clusters_means = k_means.cluster_centers_.squeeze()
    X_clustered = k_means.labels_
    X_clustered.shape = img.shape
    print(np.unique(X_clustered))
    # plt.figure(), plt.imshow(X_clustered), plt.title('Base Kmean segmentation')
    #REmOVING HOLES
    X_clustered = skm.remove_small_holes(X_clustered, remove_holes)

    X_clustered = ~(skm.remove_small_holes(~X_clustered, remove_holes))

    # First iteration
    first_img = skm.binary_closing(X_clustered, footprint = closing_fp1)

    # FILLING HOLES
    first_img = skm.remove_small_holes(first_img, remove_holes)

    # plt.figure(), plt.imshow(first_img), plt.title(f'Binary Closing with First FP')
    # Euclidean Distance of each 1 pixel to the background(0), Local Maximas are the water basins


    distance = ndi.distance_transform_edt(first_img)

    # Looking for local maximums in the footprint neighborhoods
    coords = peak_local_max(distance, footprint=fp1, labels=first_img)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=first_img)
    
    # plt.figure()
    # plt.subplot(1, 2, 1), plt.imshow(img)
    # plt.subplot(1, 2, 2), plt.imshow(-distance, cmap='gray')


    # Finding the length of each label
    SUMS = []

    # The labels that we will use in the next iteration
    new_labels = np.zeros(labels.shape)
    # First solidified labels
    first_labels = np.zeros(labels.shape)

    # Finding the maximum of the labels to iterate through all of them
    max_lab = np.max(labels)
    for i in range(1, max_lab+1):
        nth_label = np.where(labels == i, 1, 0)
        nth_map = np.where(labels == i, i, 0)
        sums = np.sum(nth_label)
        if min1 < sums < max1:
            first_labels += nth_map
        if sums > max1:
            new_labels = np.logical_or(new_labels, nth_label)
        SUMS.append(sums)
    
    # STEM OF labels and the First labels
    # plt.figure()
    # plt.subplot(1, 2, 1), plt.imshow(labels, cmap = plt.cm.nipy_spectral), plt.title('First Labels')
    # plt.subplot(1, 2, 2), plt.stem(SUMS), plt.title('#Nb of pixels in each label')

    # Second iteration
    new_labels = np.logical_and(new_labels, X_clustered)

    # Plotting the remaining labels
    # plt.figure()
    # plt.imshow(new_labels), plt.title('Remaining Regions')


    new_labels = skm.binary_closing(new_labels, footprint = closing_fp2)
    distance = ndi.distance_transform_edt(new_labels)




    # Looking for local maximums in the footprint neighborhoods
    # fp = skm.disk(90)
    coords = peak_local_max(distance, footprint=fp2, labels=new_labels)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    second_labels = watershed(-distance, markers, mask=new_labels)

    # plt.figure()
    # plt.imshow(second_labels), plt.title('Second Watershed with (90, 90) Local Max')
    # FInal
    final_lables = second_labels + first_labels
    
    final_output = np.zeros(labels.shape)
    # Removing Final small objects
    max_lab = int(np.max(final_lables))
    for i in range(1, max_lab+1):
        nth_label = np.where(final_lables == i, 1, 0)
        nth_map = np.where(final_lables == i, i, 0)
        sums = np.sum(nth_label)
        if min1/5 < sums:
            final_output += nth_map


    
    print('Num of labels', np.max(final_lables))
    
    # plt.figure()
    # plt.imshow(final_output, cmap = plt.cm.nipy_spectral)

    # plt.figure()
    # plt.imshow(final_lables, cmap = plt.cm.nipy_spectral)
    # DISK 5
    segmented_regions = find_boundaries(final_output, mode = 'thick')
    segmented_regions = skm.dilation(segmented_regions, skm.disk(1))
    
    skio.imsave(f"Labels {filename}.png", final_lables)
    final_image = raw_img.copy()
    final_image[segmented_regions == 1] = [255, 0, 0]
    if return_kmeans == True:
        return final_image, X_clustered
    else:
        return final_image