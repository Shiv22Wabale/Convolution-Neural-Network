# Emotion Recognition using ConvNet Tensorflow

Problem Statement : Video Content Analytics (VCA)

Video content analysis is the capability to automatically analyze video to detect and 
determine temporal and spatial events. These algorithms need to be implemented as a 
software component which can be easily integrated with software based MCU solutions 
which stream and record feeds from the video codec and cameras. There are many 
examples in the technology field like Video Motion Detection with regard to a fixed 
background scene and/or security examples like crowd management solutions, notably at 
The O2 Arena in London and The London Eye. However, this project requires development of 
VCA algorithms which can be integrated with an existing Tele-ICU solutions. 

Solution :

Hi, the idea for cracking this problem is very simple. The video content analytics will be performed using Tensflow machine learning architecture. As Tensflow integrate with the GPU, the near time approach is possible. With the suitable and compatible hardware the Neural Network will be build to make the VCA. In this solution the images captured by the camera are filtered through trained network. The sentiment analysis will be applied along the deep net. This sentiment analysis will provide the emotional state of the patient. The main idea behind this is to pickup the images streaming from camera and pass it through Deep Neural Network to find the probabilities of multiple temporal and spatial events. The most important part come when the number of parameter increases. This can be tackled by using the net similar to GoogLeNet. GoogLeNet, Convolutional Network from Szegedy et al. from Google was the winner of ILSVRC 2014. Its main contribution was the development of an Inception Module that dramatically reduced the number of parameters in the network. This approach would definitely help to understand the events and assign the accurate possibilities to it. The the assignment of possibilities to each event is given by the last fully connected layer. Once this probabilities are assigned, the predefined threshold would would enable alert at Tele-ICU. The Deep Network build to find the accurate result of 98% is made-up of INPUT -> [CONV -> RELU]2 -> POOL]3 -> [FC -> RELU]*2 -> FC where, INPUT : Video data CONV : Convolutional Neural Networks RELU : Rectified Linear Unit POOL : Max Pooling FC : Fully Connected Network

This choice of deep net would give the benefit to make the quicker analysis and CONV would provide less parameter due parameter sharing concept.

The model proposed to crack this problem statement is to build the neural network and make the video content analytics. The project uses Tensorflow architecture to build the deep network of 4 hidden layer, the sample images are passes through the network to make the analysis the use of high speed GPU makes this approve to be implemented in real time. The train neural network make the analysis on the validation videos. This videos would get 7 labels with the their emotion level. The challenges faced during the this project is to design and building the deep neural network. As it contains 2 fully connected layer and lots of hyper parameters. Making the correct value choice for those variable has enable system to work on. The system uses the dynamic learning algorithm to fine tune the videos in response to videos provided by the instructor or the system monitoring person. This dynamic approach help the model to grow further, give more accuracy over the time. This is how the machine learning has been cracked here.

Source : Challenge https://www.hackerearth.com/sprints/ge-healthhack-creating-a-smarter-healthier-india/
