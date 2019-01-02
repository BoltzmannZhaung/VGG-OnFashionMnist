# -*- coding: utf-8 -*-
import tensorflow as tf

def variable_L2_conv(var_name,var_shape,var_trainable,var_init,var_decay):
	var_kernel = tf.get_variable(name=var_name,
					shape=var_shape,
					dtype=tf.float32,
					trainable=var_trainable,
					initializer=tf.constant_initializer(var_init))
	if var_decay is not None:
		weight_l2 = tf.multiply(tf.nn.l2_loss(var_kernel),var_decay,name='weight_l2')
		tf.add_to_collection('total_loss',weight_l2)

	return var_kernel

def variable_L2_fc(var_name,var_shape,var_trainable,var_decay):
	var_kernel = tf.get_variable(name=var_name,
					shape=var_shape,
					dtype=tf.float32,
					trainable=var_trainable,
					initializer=tf.contrib.layers.xavier_initializer())
	if var_decay is not None:
		weight_l2 = tf.multiply(tf.nn.l2_loss(var_kernel),var_decay,name='weight_l2')
		tf.add_to_collection('total_loss',weight_l2)

	return var_kernel


def conv_layer(layer_name,input,kernel_size,out_depth,kernel_stride,w,b,no_frozen,w_decay):
	in_depth = input.shape[-1]
	with tf.name_scope(layer_name) as scope:
		kernel = variable_L2_conv(var_name=scope+"w",
							 var_shape=[kernel_size,kernel_size,in_depth,out_depth],
							 var_trainable=no_frozen,
							 var_init=w,
							 var_decay=w_decay)
		
		bias = tf.get_variable(name=scope+"b",
							shape= out_depth,
							dtype=tf.float32,
							trainable=no_frozen,
							initializer=tf.constant_initializer(b))
		
		conv_executed = tf.nn.conv2d(input,kernel,
							   (1,kernel_stride,kernel_stride,1),
							   padding='SAME')

		bias_executed = tf.nn.bias_add(conv_executed,bias)

		activation_executed = tf.nn.relu(bias_executed,name=scope)

		return activation_executed

def maxpool_layer(layer_name,input,pooling_size,pooling_stride):
	return tf.nn.max_pool(input,name=layer_name,
						  ksize=[1,pooling_size,pooling_size,1],
						  strides=[1,pooling_stride,pooling_stride,1],
						  padding='SAME')

def fc_layer(layer_name,input,out_depth,no_frozen,w_decay):
	in_depth = input.shape[-1]
	with tf.name_scope(layer_name) as scope:
		kernel = variable_L2_fc(var_name=scope+"w",
							 var_shape=[in_depth,out_depth],
							 var_trainable=no_frozen,
							 var_decay=w_decay)

		bias = tf.Variable(tf.random_normal(shape=[out_depth],stddev=0.01),name=scope+"b")

		bias_executed = tf.nn.bias_add(tf.matmul(input, kernel), bias)

		activation_executed = tf.nn.relu(bias_executed,name=scope+'relu')

		return activation_executed

def out_layer(layer_name,input,out_depth,w_decay,no_frozen):
	in_depth = input.shape[-1]
	with tf.name_scope(layer_name) as scope:
		kernel = variable_L2_fc(var_name=scope+"w",
							 var_shape=[in_depth,out_depth],
							 var_trainable=no_frozen,
							 var_decay=w_decay)

		bias = tf.Variable(tf.random_normal(shape=[out_depth],stddev=0.01),name=scope+"b")

		bias_executed = tf.nn.bias_add(tf.matmul(input, kernel), bias)

		activation_executed = tf.nn.relu(bias_executed,name=scope+'relu')

		return activation_executed



def inference(input_data,weights,dp_rate,w_decay,no_frozen=True):
	conv11 = conv_layer(layer_name='conv11',input=input_data,kernel_size=3,
						out_depth=64,kernel_stride=1,w_decay=w_decay,
						w=weights.conv11_w,b=weights.conv11_b,no_frozen=no_frozen)
	conv12 = conv_layer(layer_name='conv12',input=conv11,kernel_size=3,
						out_depth=64,kernel_stride=1,w_decay=w_decay,
						w=weights.conv12_w,b=weights.conv12_b,no_frozen=no_frozen)
	pooling1 = maxpool_layer(layer_name='pooling1',input=conv12,
							 pooling_size=2,pooling_stride=2)

	conv21 = conv_layer(layer_name='conv21',input=pooling1,kernel_size=3,
						out_depth=128,kernel_stride=1,w_decay=w_decay,
						w=weights.conv21_w,b=weights.conv21_b,no_frozen=no_frozen)
	conv22 = conv_layer(layer_name='conv22',input=conv21,kernel_size=3,
						out_depth=128,kernel_stride=1,w_decay=w_decay,
						w=weights.conv22_w,b=weights.conv22_b,no_frozen=no_frozen)
	pooling2 = maxpool_layer(layer_name='pooling2',input=conv22,
							 pooling_size=2,pooling_stride=2)

	conv31 = conv_layer(layer_name='conv31',input=pooling2,kernel_size=3,
						out_depth=256,kernel_stride=1,w_decay=w_decay,
						w=weights.conv31_w,b=weights.conv31_b,no_frozen=no_frozen)
	conv32 = conv_layer(layer_name='conv32',input=conv31,kernel_size=3,
						out_depth=256,kernel_stride=1,w_decay=w_decay,
						w=weights.conv32_w,b=weights.conv32_b,no_frozen=no_frozen)
	conv33 = conv_layer(layer_name='conv33',input=conv32,kernel_size=3,
						out_depth=256,kernel_stride=1,w_decay=w_decay,
						w=weights.conv33_w,b=weights.conv33_b,no_frozen=no_frozen)
	pooling3 = maxpool_layer(layer_name='pooling3',input=conv33,
							 pooling_size=2,pooling_stride=2)


#For Fashion-MNIST dataset,3 convolutional layers are enough.
#
#	conv41 = conv_layer(layer_name='conv41',input=pooling3,kernel_size=3,
#						out_depth=512,kernel_stride=1,w_decay=w_decay,
#						w=weights.conv41_w,b=weights.conv41_b,no_frozen=no_frozen)
#	conv42 = conv_layer(layer_name='conv42',input=conv41,kernel_size=3,
#						out_depth=512,kernel_stride=1,w_decay=w_decay,
#						w=weights.conv42_w,b=weights.conv42_b,no_frozen=no_frozen)
#	conv43 = conv_layer(layer_name='conv43',input=conv42,kernel_size=3,
#						out_depth=512,kernel_stride=1,w_decay=w_decay,
#						w=weights.conv43_w,b=weights.conv43_b,no_frozen=no_frozen)
#	pooling4 = maxpool_layer(layer_name='pooling4',input=conv43,
#							 pooling_size=2,pooling_stride=2)
#
#	conv51 = conv_layer(layer_name='conv51',input=pooling4,kernel_size=3,
#						out_depth=512,kernel_stride=1,w_decay=w_decay,
#						w=weights.conv51_w,b=weights.conv51_b,no_frozen=no_frozen)
#	conv52 = conv_layer(layer_name='conv52',input=conv51,kernel_size=3,
#						out_depth=512,kernel_stride=1,w_decay=w_decay,
#						w=weights.conv52_w,b=weights.conv52_b,no_frozen=no_frozen)
#	conv53 = conv_layer(layer_name='conv53',input=conv52,kernel_size=3,
#						out_depth=512,kernel_stride=1,w_decay=w_decay,
#						w=weights.conv53_w,b=weights.conv53_b,no_frozen=no_frozen)
#	pooling5 = maxpool_layer(layer_name='pooling5',input=conv53,
#							 pooling_size=2,pooling_stride=2)



	shape_layer3 = pooling3.get_shape()
	shape_flattened = shape_layer3[1].value * shape_layer3[2].value * shape_layer3[3].value
	#print('Debug INFO: The length is %d'%(shape_flattened))
	result_reshaped = tf.reshape(pooling3,[-1,shape_flattened],name='flatten')

	fc6 = fc_layer(layer_name='fc6',input=result_reshaped,out_depth=4096,
				   w_decay=w_decay,no_frozen=no_frozen)
	fc6_DP = tf.nn.dropout(fc6,dp_rate,name='fc6_dropout')

	fc7 = fc_layer(layer_name='fc7',input=fc6_DP,out_depth=4096,
				   w_decay=w_decay,no_frozen=no_frozen)
	fc7_DP = tf.nn.dropout(fc7,dp_rate,name='fc7_dropout')

	fc8 = out_layer(layer_name='fc8',input=fc7_DP,out_depth=10,
					w_decay=w_decay,no_frozen=no_frozen)

	out_softmax = tf.nn.softmax(fc8)

	#prediction = tf.argmax(out_softmax,1)
	#prediction = tf.cast(prediction,tf.float32)
	
	return out_softmax


def loss(prediction,labels):
	labels = tf.cast(labels,tf.int64)

	shape_prediction = prediction.get_shape()
	shape_labels = labels.get_shape()
	print('shape_prediction:%s'%(shape_prediction))
	print('shape_labels:%s'%(shape_labels))

	#logits=tf.argmax(prediction, 1)

	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,
																   labels=labels,
																   name='cross_entropy')
	cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy_mean')
	
	tf.add_to_collection('total_loss',cross_entropy_mean)

	loss_all = tf.add_n(tf.get_collection('total_loss'),name='loss_all')

	return loss_all









