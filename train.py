import os
import time
import numpy as np
import tensorflow as tf
import preprocess as pre
from components import inference,loss
from Weights import Weights_Tranined

IMG_SIZE = 28
IMG_CHANNEL = 3
CLASS_NUM = 10
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.5
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0005
TRAINING_STEP = 5001
DROPOUT_RATE = 0.5


def train_model(t_x,t_y,weights,dprate,imgsize,imgchannel,batchsize,train_step,
				learningrate,learningdecay,regurate):
	
	x = tf.placeholder(tf.float32,[None,imgsize,imgsize,imgchannel],name='x_input')
	y_real = tf.placeholder(tf.float32, [None,], name='y_real')

	y_hat = inference(input_data=x, weights=weights , dp_rate=dprate, w_decay=regurate, no_frozen=True)

	cost = loss(prediction=y_hat,labels=y_real)

	optimizer = tf.train.RMSPropOptimizer(learning_rate=learningrate,
							decay=learningdecay).minimize(cost)

	correct_prediction = tf.equal(tf.cast(tf.argmax(y_hat, 1),tf.float32), y_real)

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


	TRAINING_NUMBERS = len(t_x)

	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		print('Training begin at:',time.strftime('%c',time.localtime()))

		for i in range(train_step):
			for batch in range(TRAINING_NUMBERS//batchsize):
				batch_x = t_x[batch*batchsize:min((batch+1)*batchsize,TRAINING_NUMBERS)]
				batch_y = t_y[batch*batchsize:min((batch+1)*batchsize,TRAINING_NUMBERS)]

				losses = sess.run(cost, feed_dict={x: batch_x, y_real: batch_y})
				opt = sess.run(optimizer, feed_dict={x: batch_x, y_real: batch_y})
				acc = sess.run(accuracy, feed_dict={x: batch_x, y_real: batch_y})

			print('----------------------------------------------------------------')
			print('Iterative times :',str(i),'Loss ={:.6f}'.format(losses),\
                      'Training Accuracy= {:.5f}'.format(acc))
			print("ONE Time Iteration Finished!")
		print('Training done at:',time.strftime('%c',time.localtime()))
				


def main(arg=None):
	#load npz file.
	weights = np.load('./vgg16_weights.npz')

	#Weights_Tranined store the data of vgg16_weights.npz that download from internet.
	#For example: As a member of Weights_Tranined, 'conv11_w' corresponds with the 
	#			 weights of the first conv3-64. 'conv11_b' corresponds with the bias
	#			 of the first conv3-64.
	w_trained = Weights_Tranined(weights)

	#Build training and validation sets
	data_path = os.path.join(os.getcwd(),'fmnist')
	print(data_path)
	imgs,labels = pre.processing(data_path,CLASS_NUM)
	train_img,train_label,validation_img,validation_label = pre.split(imgs,labels)

	

	print(train_img.shape)
	print(train_label.shape)

	train_model(t_x=train_img,t_y=train_label,weights=w_trained,dprate=DROPOUT_RATE,
				imgsize=IMG_SIZE,imgchannel=IMG_CHANNEL,
				batchsize=BATCH_SIZE,train_step=TRAINING_STEP,learningrate=LEARNING_RATE_BASE,
				learningdecay=LEARNING_RATE_DECAY,regurate=REGULARIZATION_RATE)


if __name__ == "__main__":
    # execute only if run as a script
    main()




