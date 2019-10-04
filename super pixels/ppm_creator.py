'''
This is a script to test the ppm - p3 writting.

'''
import imageio
import array
import cv2

'''
# PPM header
width = 250
height = 250
maxval = 255
ppm_header = f'P6 {width} {height} {maxval}\n'
 
# PPM image data (filled with blue)
image = array.array('B', [0, 0, 255] * width * height)
 
# Save the PPM image as a binary file
with open('blue_example.ppm', 'wb') as f:
    #uncomment this to write as a p6 format
    f.write(bytearray(ppm_header, 'ascii'))
    
    image.tofile(f)


alpha =cv2.imread('E:/SENZMATE/my work/submission/week 4/blue_example.ppm');
cv2.imshow('blue_example.ppm',alpha);
'''

im=cv2.imread('E:\SENZMATE\my work\submission\week 3\hair segmentation- reference\gloc\gloc\data\lfw_funneled\Inga_Hall\Inga_Hall_0001.jpg');
#cv2.imshow('image',im);
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB);

print(im.dtype);
imageio.imwrite('IngaHall.ppm', im, format='PPM-FI', flags=1);

