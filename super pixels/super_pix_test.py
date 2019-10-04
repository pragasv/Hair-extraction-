# import the necessary packages
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio

'''
This segment is the test on super pixel and their labeling together with numbering 

slic(image, n_segments=100, compactness=10.0, max_iter=10, sigma=0, spacing=None, multichannel=True, convert2lab=None,
enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False)[source]
Segments image using k-means clustering in Color-(x,y,z) space.

'''


### --- data directories --- ###
data_dir = "E:\SENZMATE\my work\submission\week 3\hair segmentation- reference\gloc\gloc\data";

## LFW funneled images
lfw_dir = "E:\SENZMATE\my work\submission\week 3\hair segmentation- reference\gloc\gloc\data\lfw_funneled";

'''
## segmentation labels stored as images
label_dir = "E:\SENZMATE\my work\submission\week 3\hair segmentation- reference\gloc\gloc\data\parts_lfw_funneled_gt_images";

## superpixel data
spmat_dir = "E:\SENZMATE\my work\submission\week 3\hair segmentation- reference\gloc\gloc\data\parts_lfw_funneled_superpixels_mat";

## ground truth for the superpixels
gt_dir = "E:\SENZMATE\my work\submission\week 3\hair segmentation- reference\gloc\gloc\data\parts_lfw_funneled_gt";

## node and edge features
features_dir = "E:\SENZMATE\my work\submission\week 3\hair segmentation- reference\gloc\gloc\data\parts_lfw_funneled_spseg_features";
'''

im = cv2.imread(lfw_dir+"\Inga_Hall\Inga_Hall_0001.jpg");


##breaking into super pixels and marking boundary

#change this values and see
#initial values used n_segments=255,sigma=0.4,compactness=24,max_iter=100

segments = slic(im,n_segments=255,sigma=0.4,compactness=24,max_iter=100 );
im2 = np.reshape(segments,(im.shape[1],im.shape[0]));
im3=mark_boundaries(cv2.cvtColor(im,cv2.COLOR_BGR2RGB), im2)


#im2 = cv2.cvtColor(im2,cv2.COLOR_RGB2BGR);
#cv2.imshow('segmented and boundary marked',im2);

plt.figure();
plt.imshow(im3);
plt.show();




#labeling the super pixel cluster heads / ideantification of unique pixel heads
'''
# loop over the unique segment values
for (i, segVal) in enumerate(np.unique(segments)):
	# construct a mask for the segment
	print ("[x] inspecting segment %d" % (i+1))
	mask = np.zeros(im.shape[:2], dtype = "uint8")
	mask[segments == segVal] = 255

	cv2.imshow("Mask", mask)
	#cv2.waitKey(0)  #uncomment to see the labeling clearly 
'''
        
##mask should be converted to be from 1-255 to be saved
##to create the sp_mat_dir

#im2 is the segmented boundary

im2 = cv2.cvtColor(np.uint8(im2),cv2.COLOR_GRAY2RGB);
im2 = im2 +1;
imageio.imwrite('IngaHall.ppm', im2, format='PPM-FI', flags=1);

#print(im2);

#we dont merge the small pieces and renumber the large ones 

