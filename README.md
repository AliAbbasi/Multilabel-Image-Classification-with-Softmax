This is an example of Multi-label Softmax Classifier written by python and tensorflow.

Pre-requestes:

Python 2.7.11
Tensorflow 1.1
Numpy
PIL



The 'raw_images' directory shows the dataset include two labeled images of objects and shapes.

The dataset consist of 8 category trivial objects include: helmet, kettle, joystick, keyboard, mouse, stapler, barrel and mug
and small shapes inside each image include: triangle, square and circle. 

The first labels show the object category and the second label shows the shape inside each image.

![alt text](https://github.com/AliAbbasi/Multilabel-Image-Classification-with-Softmax/blob/master/raw_images/0a%20(2).png)
![alt text](https://github.com/AliAbbasi/Multilabel-Image-Classification-with-Softmax/blob/master/raw_images/1a%20(2).png)
![alt text](https://github.com/AliAbbasi/Multilabel-Image-Classification-with-Softmax/blob/master/raw_images/2b%20(2).png)
![alt text](https://github.com/AliAbbasi/Multilabel-Image-Classification-with-Softmax/blob/master/raw_images/3c%20(2).png)

Use 'dataAugmentation.py' to create artificially more image by rotating them. you can also add flip horizontally and vertically and more rotation degrees, also, use blur, add noise, transfer pixels to right, left, up, down, zoom in, zoom out and other augmentation methods.

The original images is 200x200 pixels, use 'resize.py' to resize images to any size you want. The classifier work with 32x32 input images.

'convertToMatrix.py' convert all images to '.npz' file and add ground truth label to them.
The name of images shows the labels. first character shows the index of 1 in one-hot vector format, and the second character stands for the shape inside each image, 'a' for squares, 'b' for triangles and 'c' for circles.

'32x32-two-labeled-images.npz' file is created by me based on ~12k augmented images.




Contact me if you have any question or suggestion 

email: abbasi.ali.tab@gmail.com

Linkedin: https://www.linkedin.com/in/ali-abbasi-6b5512b7/
