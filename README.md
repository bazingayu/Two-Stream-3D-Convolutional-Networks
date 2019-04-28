# Two-Stream-3D-Convolutional-Networks
Two-Stream 3D Convolutional Networks, merge simple VGG16 with two_steam 3D convolution network
in preprocessing tfrecord_biovid, you can convert your data to tfrecords.
then you can train the network by train.py
model3.py is our two steam with vgg convolution networks inference programs.
read.py is the precess of reading and shuffle data.

extract_flow2 contains the program to precess your raw sequences(m frames) to flow sequences(m-1 frames)
extract_face_canny.py is our try to extract the face canny feature and then extract flows for canny data. It's turns out that it's invalid for training.

Due to our computional resources, our VGG16's channel are less than the original version. so you can change it by yourself. 
