
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import datetime
import inception
import math
import scipy
import scipy.misc
import glob

inception.data_dir = 'inception/'

inception.maybe_download()
model = inception.Inception()
resized_image = model.resized_image
y_pred = model.y_pred
y_logits = model.y_logits

# Set the graph for the Inception model as the default graph,
# so that all changes inside this with-block are done to that graph.
with model.graph.as_default():
    # Add a placeholder variable for the target class-number.
    # This will be set to e.g. 300 for the 'bookcase' class.
    pl_cls_target = tf.placeholder(dtype=tf.int32)

    # Add a new loss-function. This is the cross-entropy.
    # See Tutorial #01 for an explanation of cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits, labels=[pl_cls_target])

    # Get the gradient for the loss-function with regard to
    # the resized input image.
    gradient = tf.gradients(loss, resized_image)

session = tf.Session(graph=model.graph)
time = datetime.datetime.now().strftime('%m%d%H%M%S')

images = glob.glob("./images/*.JPEG")

# Parameter configs
cls_target=300
noise_limit=3.0
required_score=0.99
show_image=False

result_file = open('result.txt', 'w')



def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


# In[ ]:


# Check if the image is still classified as original class
def test_precision(i, image):
    test_image = np.clip(a=image, a_min=0.0, a_max=255.0)

    # Create a feed-dict. This feeds the noisy image to the
    # tensor in the graph that holds the resized image, because
    # this is the final stage for inputting raw image data.
    # This also feeds the target class-number that we desire.
    feed_dict = {model.tensor_name_resized_image: [test_image],
                 pl_cls_target: cls_target}

    # Calculate the predicted class-scores as well as the gradient.
    pred, grad = session.run([y_pred, gradient],
                             feed_dict=feed_dict)

    final_class = np.argmax(pred)
    '''
    
    # Names for the source and target classes.
    name_source = model.name_lookup.cls_to_name(final_class,
                                                only_first_name=True)
    print('is classified as {} with score {}'. format(name_source, pred.max()))
    '''
    
    # Convert the predicted class-scores to a one-dim array.
    pred = np.squeeze(pred)

    # The scores (probabilities) for the source and target classes.
    score_source = pred[cls_source]
    score_target = pred[cls_target]

    return score_source > score_target, final_class == cls_source


# In[ ]:


# Defense: TV compression
# Simple heuristic to implement total variance
def tv_compress(image):
    lambda_tv = 0.5
    bernoulli_p = 0.25
    A = image
    # TODO: make it stop at diff < threshold instead of 100 iterations
    for x in range(100):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.random.random() > 0.25:
                    continue
                cnt_ = 0
                sum_ = 0
                # Self cell
                cnt_ += 1
                sum_ += A[i, j]
                # Adjecent cells
                if(i > 0):
                    cnt_ += lambda_tv
                    sum_ += A[i-1, j] * lambda_tv
                if(j > 0):
                    cnt_ += lambda_tv
                    sum_ += A[i, j-1] * lambda_tv
                if(i < A.shape[0]-1):
                    cnt_ += lambda_tv
                    sum_ += A[i+1, j] * lambda_tv
                if(j < A.shape[1]-1):
                    cnt_ += lambda_tv
                    sum_ += A[i, j+1] * lambda_tv
                A[i, j] = sum_ / cnt_
    return A


# In[ ]:


# Defense: bit compression
def bit_compress(image, compress_bit):
    compressed_image = image
    for i in range(compressed_image.shape[0]):
        for j in range(compressed_image.shape[1]):
            for k in range(compressed_image.shape[2]):
                compressed_image[i,j,k] -= compressed_image[i,j,k]% math.pow(2, compress_bit)
    return compressed_image


# In[ ]:


# Defense: kmeans
def find_nearest_cluster(current_pixel, k_means):
	diff_sq = np.square(k_means - np.tile(current_pixel,(16,1)))
	sum_of_sq = diff_sq[:,0] + diff_sq[:,1] + diff_sq[:,2]
	return np.argmin(sum_of_sq)

def kmeans_compress(image):
# (b) Calculate 16 means' centroid
    A = image
    k = 16
# Random initialize each k-mean's centroid from a cell in small picture
    index1 = np.random.randint(0,A.shape[0],k)
    index2 = np.random.randint(0,A.shape[1],k)
    k_means = A[index1, index2, :].astype(float)
    for x in range(100):
        new_means = np.zeros((k, A.shape[2])).astype(float)
        new_means_count = np.zeros(k).astype(float)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                nearest_centroid = find_nearest_cluster(A[i,j,:], k_means)
                new_means_count[nearest_centroid] += 1
                new_means[nearest_centroid] += A[i,j,:]
	# Replace the k-means with new centroids
        new_means_count[new_means_count==0] = 1
        k_means = np.divide(new_means,np.tile(new_means_count, (3,1)).T)

# (c) Replace large image's all color with the nearest centroid's color
    compressed_image = image
    for i in range(compressed_image.shape[0]):
        for j in range(compressed_image.shape[1]):
            nearest_centroid = find_nearest_cluster(compressed_image[i,j,:], k_means)
            compressed_image[i,j,:] = k_means[nearest_centroid]
        
    return compressed_image


# In[ ]:


# Defense: spatial smoothing, right now using 3X3
def median(lst):
    quotient, remainder = divmod(len(lst), 2)
    if remainder:
        return sorted(lst)[quotient]
    return sum(sorted(lst)[quotient - 1:quotient + 1]) / 2.

def checkindex(image, index_x, index_y):
    if(index_x < 0 or index_x >= image.shape[0]):
        return False;
    if(index_y < 0 or index_y >= image.shape[1]):
        return False;
    return True;

def find_median_in_sliding_windown(image, i, j, k, m, n):
    list = []
    for x in range(-m+1, m):
        for y in range(-n+1, n):
            if(checkindex(image, i+x, j+y)):
                list.append(image[i+x, j+y, k])
    return median(list)
            
# m = n = 1 is just using the pixel itself, i.e. no change
def spatial_smoothing(image, m, n):
    compressed_image = image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                compressed_image[i, j, k] = find_median_in_sliding_windown(image, i, j, k, m, n)
    return compressed_image

#Experiment on bit compression with Deepfool
total = len(images)
success1 = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
success2 = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
success3 = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
# If it remained in same class
precision1 = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
precision2 = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
precision3 = [0., 0., 0., 0., 0., 0., 0., 0., 0.]

threshold = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
for image in images:
    print (datetime.datetime.now())
    feed_dict = model._create_feed_dict(image_path=image)

    #image = img[100]
    #image_path= 'cifar/'
    #feed_dict = model._create_feed_dict(image=image)



    pred, image = session.run([y_pred, resized_image],
                                  feed_dict=feed_dict)




    cls_source = np.argmax(pred)
    cls_target = 300

    # Score for the predicted class (aka. probability or confidence).
    score_source_org = pred.max()

    # Names for the source and target classes.
    name_source = model.name_lookup.cls_to_name(cls_source,
                                                only_first_name=True)
    name_target = model.name_lookup.cls_to_name(cls_target,
                                                only_first_name=True)

    # Initialize the noise to zero.
    noise = np.zeros(image[0].shape)
    iterations = 0
    # Perform a number of optimization iterations to find
    # the noise that causes mis-classification of the input image.
    index = 0

    # Initailization for classless Deepfool parameters
    #pert = np.inf
    #w = np.squeeze(np.zeros(image[1:]))

    for i in range(10000):
        iterations = i

        # The noisy image is just the sum of the input image and noise.
        noisy_image = image + noise

        # Ensure the pixel-values of the noisy image are between
        # 0 and 255 like a real image. If we allowed pixel-values
        # outside this range then maybe the mis-classification would
        # be due to this 'illegal' input breaking the Inception model.
        noisy_image = np.clip(a=noisy_image, a_min=0.0, a_max=255.0)

        # Create a feed-dict. This feeds the noisy image to the
        # tensor in the graph that holds the resized image, because
        # this is the final stage for inputting raw image data.
        # This also feeds the target class-number that we desire.
        feed_dict_source = {model.tensor_name_resized_image: noisy_image,
                     pl_cls_target: cls_source}

        feed_dict_target = {model.tensor_name_resized_image: noisy_image,
                     pl_cls_target: cls_target}

        # Calculate the predicted class-scores as well as the gradient.
        pred, grad_src = session.run([y_pred, gradient],
                                 feed_dict=feed_dict_source)
        pred, grad_tgt = session.run([y_pred, gradient],
                                 feed_dict=feed_dict_target)

        # Convert the predicted class-scores to a one-dim array.
        pred = np.squeeze(pred)

        # The scores (probabilities) for the source and target classes.
        score_source = pred[cls_source]
        score_target = pred[cls_target]

        # Squeeze the dimensionality for the gradient-array.
        w_k = np.array(grad_tgt).squeeze() - np.array(grad_src).squeeze()
        f_k = score_target - score_source
        pert_k = (abs(f_k) + 1e-80) * w_k / np.linalg.norm(w_k)
        #print(pert_k)

        test_precision(iterations, (image + noise)[0])

        l2_norm = math.sqrt(np.linalg.norm(noise)/np.linalg.norm(image))
        step_size = 0.02/min(0.01, np.linalg.norm(pert_k)) + l2_norm*l2_norm/0.0001
        print ('l2 norm is {}'.format(l2_norm))

        # If the score for the target-class is not high enough.
        if index < len(threshold):
            #if score_target < required_score and index < len(threshold):
            # Update the image-noise by subtracting the gradient
            # scaled by the step-size.
            noise += step_size * pert_k

            # Ensure the noise is within the desired range.
            # This avoids distorting the image too much.
            noise = np.clip(a=noise,
                            a_min=-noise_limit,
                            a_max=noise_limit)
            '''
            if (iterations % 10 == 0):
                print("Print defense effect")
                # Chose whatever defense method you want to use on the bottom.
                test_precision(iterations, spatial_smoothing((image + noise)[0], 3, 3))
            '''

            if l2_norm >= threshold[index]:
                #print("inside while loop")
                # Abort the optimization because the score is high enough.
                x_1, x_2 = test_precision(iterations, tv_compress((image + noise)[0]))
                success1[index] += x_1
                precision1[index] += x_2

                if(x_1!=0):
                    #print("index is ", index)
                    index += 1
                else:
                    index += 10

        else:
            print(success1)
            print(precision1)

            break;

    print("finished image ")


