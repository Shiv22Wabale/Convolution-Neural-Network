import cv2
import sys
import os
from numpy import array
import tensorflow as tf
import numpy as np
from random import randint

from sklearn.preprocessing import normalize

from Tkinter import *
import tkMessageBox
import Tkinter

checkpoint_steps = 200
train = 0 # Training or restoring
checkpoint_dir = '/home/shivraj/face_detection/saveTensorFlow3/'
learn = 0 
exit = 1
validation = 0

def PrintCommand(the_number):
	if str(the_number) == "8" :
		global exit
		exit = 0
		return
	global validation
	global learn
	learn = 1
	validation = int(the_number)

def imp(root, pred) :
	pred[0] = pred[0].clip(min=0)
	norm = [float(i) * 100 /sum(pred[0]) for i in pred[0]]
	# True
	#print(norm)

	sort = sorted(range(len(norm)), key=lambda x: norm[x])[-2:]

	#Entry1 = Entry(root, bd =1)
	#Entry1.pack(side=TOP,padx=5,pady=5)
	root.title("Patient")
	root.geometry("300x1500")

	tmp = str(norm[0])[:4]
	tmp += "%  Neutral"
	bttn1 = Tkinter.Button(root, text =tmp, command = lambda: PrintCommand(0), bg="red" if 0 in sort else "blue")
	bttn1.pack(side = TOP,padx=10,pady=20)


	tmp = str(norm[1])[:4]
	tmp += "%  Anger"
	bttn2 = Tkinter.Button(root, text =tmp, command = lambda: PrintCommand(1), bg="red" if 1 in sort else "blue")
	bttn2.pack(side = TOP,padx=10,pady=20)

	tmp = str(norm[2])[:4]
	tmp += "%  Contempt"
	bttn3 = Tkinter.Button(root, text =tmp, command = lambda : PrintCommand(2), bg="red" if 2 in sort else "blue")
	bttn3.pack(side = TOP,padx=10,pady=20)

	tmp = str(norm[3])[:4]
	tmp += "%  Disgust"
	bttn4 = Tkinter.Button(root, text =tmp, command = lambda : PrintCommand(3), bg="red" if 3 in sort else "blue" )
	bttn4.pack(side = TOP,padx=10,pady=20)

	tmp = str(norm[4])[:4]
	tmp += "%  Fear"
	bttn5 = Tkinter.Button(root, text =tmp, command = lambda : PrintCommand(4), bg="red" if 4 in sort else "blue" )
	bttn5.pack(side = TOP,padx=10,pady=20)

	tmp = str(norm[5])[:4]
	tmp += "%  Happy"
	bttn6 = Tkinter.Button(root, text =tmp, command = lambda : PrintCommand(5), bg="red" if 5 in sort else "blue" )
	bttn6.pack(side = TOP,padx=10,pady=20)

	tmp = str(norm[6])[:4]
	tmp += "%  Sad"
	bttn7 = Tkinter.Button(root, text =tmp, command = lambda : PrintCommand(6), bg="red" if 6 in sort else "blue" )
	bttn7.pack(side = TOP,padx=10,pady=20)

	tmp = str(norm[7])[:4]
	tmp += "%  Distress"
	bttn8 = Tkinter.Button(root, text =tmp, command = lambda : PrintCommand(7), bg="red" if 7 in sort else "blue" )
	bttn8.pack(side = TOP,padx=10,pady=20)

	bttn9 = Tkinter.Button(root, text ="Exit", command = lambda : PrintCommand(8), bg="blue" )
	bttn9.pack(side = TOP,padx=10,pady=20)



def absoluteFilePaths(directory):
	for dirpath,_,filenames in os.walk(directory):
		for f in filenames:
			yield os.path.abspath(os.path.join(dirpath, f))


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.2, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


cascPath = sys.argv[1]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

#Build tensorflow deep neural 
x_image = tf.placeholder(tf.float32, [1, 512,  512, 3])
y_ = tf.placeholder(tf.float32, [1, 8])

# Reshape the image
x_image = tf.reshape(x_image, [-1, 512, 512, 3])

#--------1
W_conv1 = weight_variable([3, 3, 3, 8])
b_conv1 = bias_variable([8])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) # 128 * 128 * 8
#--------1

#---------2
W_conv2 = weight_variable([3, 3, 8, 16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # 64 * 64 * 16 
#---------2

#---------3
W_conv3 = weight_variable([5, 5, 16, 32])
b_conv3 = bias_variable([32])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3) # 32 * 32 * 32 --- 
#---------3

#---------4 fc
W_fc1 = weight_variable([32 * 32 * 32 * 4, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 32 * 32 * 32 * 4])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
#--------4 fc

#--------5 fc last
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 8])
b_fc2 = bias_variable([8])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#-------5 fc last

#error = tf.reduce_mean(tf.abs(y_conv - y_))
error = cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(error)


saver = tf.train.Saver() # to save the model

i = 0
#	Start the tensorflow session
with tf.Session() as sess :
	sess.run(tf.global_variables_initializer())
	if train == 1 :
		for step in range(20) :
			print("Step:")
			print(step)
			files=absoluteFilePaths('/home/shivraj/Pictures/Conh/Emotion')
			for f in files :
				with open(f, 'r') as rf:
					for line in rf:
						label = eval(line.replace('\n',''))
						#print label

				# Image Path
				imagePath = f.replace('Emotion', 'cohn-kanade-images').replace('_emotion.txt','.png')
				#print(f)

				#To train on randon dataset
				if randint(0,9) == 0 :
					continue


				# Get user supplied values
				#imagePath = f
				
				# Read the image
				image = cv2.imread(imagePath)
				detect = image
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

				# Detect faces in the image
				faces = faceCascade.detectMultiScale (
					gray,
					scaleFactor=1.1,
					minNeighbors=5,
					minSize=(30, 30),
					flags = cv2.cv.CV_HAAR_SCALE_IMAGE
				)

				#print("Found {0} faces!".format(len(faces)))
				#print(imagePath)
				
				crop_img = []
				# Draw a rectangle around the faces
				for (x, y, w, h) in faces:
					crop_img = detect[y: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
					# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
					crop_img = cv2.resize(crop_img, (512,512), interpolation = cv2.INTER_AREA)
					crop_img = array(crop_img).reshape(1, 512, 512, 3)

					#l = [label, label, label, label]
					#print(label)
					#index_offset = label
					labels_one_hot = [0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07]
					labels_one_hot[int(label)] = 0.51

					labels_one_hot = array(labels_one_hot).reshape(1, 8)
					print(labels_one_hot)
					train_step.run(feed_dict={x_image : crop_img, y_: labels_one_hot, keep_prob: 0.8})
					#train_accuracy = y_conv.eval(feed_dict={x_image : crop_img, y_: label, keep_prob: 1.0})
					if i % 20 == 0 : 
						print(y_conv.eval(feed_dict={x_image:crop_img, y_ : labels_one_hot, keep_prob : 1.0}))
				#		print(labels_one_hot)
					if i % checkpoint_steps == 0 :
						saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1)
					i = i + 1
		saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1)
	else :
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("Restored the network");
			exit = 1
			while(exit) :
				print(exit) 
				if learn == 1:
					for step in range(10) :
						print("Step:")
						print(step)
						files=absoluteFilePaths('/home/shivraj/Pictures/Conh/Emotion')
						for f in files :
							with open(f, 'r') as rf:
								for line in rf:
									label = eval(line.replace('\n',''))
									#print label
						#	if label == 7 :
						#		break
							# Image Path
							imagePath = f.replace('Emotion', 'cohn-kanade-images').replace('_emotion.txt','.png')
							#print(f)
							#To train on randon dataset
							if randint(0,9) == 0 :
								continue
	
							# Get user supplied values
							#imagePath = f


							#imagePath = sys.argv[1]
							# Read the image
							image = cv2.imread(imagePath)
							detect = image
							gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

							# Detect faces in the image
							faces = faceCascade.detectMultiScale (
								gray,
								scaleFactor=1.1,
								minNeighbors=5,
								minSize=(30, 30),
								flags = cv2.cv.CV_HAAR_SCALE_IMAGE
							)

							#print("Found {0} faces!".format(len(faces)))
							#print(imagePath)
							
							#crop_img = [][]
							# Draw a rectangle around the faces
							for (x, y, w, h) in faces:
								crop_img = detect[y: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
								# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
								crop_img = cv2.resize(crop_img, (512,512), interpolation = cv2.INTER_AREA)
								crop_img = array(crop_img).reshape(1, 512, 512, 3)

								#l = [label, label, label, label]
								#print(label)
								#index_offset = label
								labels_one_hot = [0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07]
								labels_one_hot[int(label)] = 0.51
								labels_one_hot = array(labels_one_hot).reshape(1, 8)
								print(labels_one_hot)
								#for _ in range (10) :
								train_step.run(feed_dict={x_image : crop_img, y_: labels_one_hot, keep_prob: 0.8})
								if i % 20 == 0 : 
									print(y_conv.eval(feed_dict={x_image:crop_img, y_ : labels_one_hot, keep_prob : 1.0}))
								if i % checkpoint_steps == 0 :
									saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1)
								i = i + 1
					saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1)
					exit = 0
				else :
					imagePath = sys.argv[2]
					files=absoluteFilePaths(imagePath)
					add = [[0,0,0,0,0,0,0,0]]
					for f in files :
						image = cv2.imread(f)
						detect = image
						gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

						# Detect faces in the image
						faces = faceCascade.detectMultiScale (
							gray,
							scaleFactor=1.1,
							minNeighbors=5,
							minSize=(30, 30),
							flags = cv2.cv.CV_HAAR_SCALE_IMAGE
						)

						#print("Found {0} faces!".format(len(faces)))
						#print(imagePath)
						image_display = image
						crop_img = []
						# Draw a rectangle around the faces
						for (x, y, w, h) in faces:
							cv2.rectangle(image_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
							#print(w, h)
							crop_img = detect[y: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
							crop_img = cv2.resize(crop_img, (512,512), interpolation = cv2.INTER_AREA)
							crop_img = array(crop_img).reshape(1, 512, 512, 3)

							pred = y_conv.eval(feed_dict={x_image:crop_img, keep_prob : 1.0})
							print(pred)
						#add = tf.accumulate_n([add, pred], shape=[1, 8], tensor_dtype=tf.int32)
					add = np.add(pred, add)
					root = Tkinter.Tk()
					Lbl1 = Label(root, text="GE HealthHack")
					Lbl1.pack(side=TOP,padx=5,pady=5)

					imp(root, array(add))
					cv2.imshow("Faces found", image_display)
					ch = cv2.waitKey()

					if ch == 27:#Escape
						cv2.destroyAllWindows()
					root.mainloop()
					if learn == 1 :
						imagePath = sys.argv[2]
						files=absoluteFilePaths(imagePath)
						add = [[0,0,0,0,0,0,0,0]]
						for f in files :
							image = cv2.imread(f)
							detect = image
							gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

							# Detect faces in the image
							faces = faceCascade.detectMultiScale (
								gray,
								scaleFactor=1.1,
								minNeighbors=5,
								minSize=(30, 30),
								flags = cv2.cv.CV_HAAR_SCALE_IMAGE
							)

							#print("Found {0} faces!".format(len(faces)))
							#print(imagePath)
							image_display = image
							crop_img = []
							# Draw a rectangle around the faces
							for (x, y, w, h) in faces:
								cv2.rectangle(image_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
							#	print(w, h)
								crop_img = detect[y: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
								crop_img = cv2.resize(crop_img, (512,512), interpolation = cv2.INTER_AREA)
								crop_img = array(crop_img).reshape(1, 512, 512, 3)
								labels_one_hot = [0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]
								labels_one_hot[int(validation)] = 0.79
								labels_one_hot = array(labels_one_hot).reshape(1, 8)
								print(labels_one_hot)	
								train_step.run(feed_dict={x_image : crop_img, y_: labels_one_hot, keep_prob: 0.8})
								pred = y_conv.eval(feed_dict={x_image:crop_img, keep_prob : 1.0})
								print(pred)
							add = np.add(pred, add)	
						root = Tkinter.Tk()

						Lbl1 = Label(root, text="GE HealthHack")
						Lbl1.pack(side=TOP,padx=5,pady=5)
						imp(root, array(add))
						root.mainloop()
						#norm = [float(i) * 100 /sum(pred[0]) for i in pred[0]]
	# True
	#print(norm)

						#print(norm)
						saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=int(i)+1)
						exit = 0

		else:
			print('...no checkpoint found...')
