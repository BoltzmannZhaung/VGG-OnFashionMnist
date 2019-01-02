import os
import gzip
import numpy as np

def load_mnist(path, kind):
	labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'%(kind))
	images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'%(kind))
	
	with gzip.open(labels_path, 'rb') as lbpath:
		labels = np.frombuffer(lbpath.read(),dtype=np.uint8,offset=8)

	with gzip.open(images_path, 'rb') as imgpath:
		images = np.frombuffer(imgpath.read(),dtype=np.uint8,
							   offset=16).reshape(len(labels), 784)
	return images, labels

def channel_1To3(images):
	index = images.shape[0]
	channel = 3
	new_images = np.zeros((index,28,28,channel), dtype=np.uint8)
	for x in range(index):
		img = (images[x]).reshape(28,28,1)
		new_images[x] = np.broadcast_to(img,(28,28,3))

	return new_images


def do_OneHot(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot


def processing(path,kinds_num,kind='train',one_hot=False):
	imgs,labels = load_mnist(path,kind)
	new_imgs = channel_1To3(imgs)
	if one_hot:
		new_labels = do_OneHot(labels,kinds_num)
		return new_imgs,new_labels
	else:
		return new_imgs,labels

def split(images,labels):
	train_images = []
	train_labels = []
	validation_images = []
	validation_labels = []

	state= np.random.get_state()
	np.random.shuffle(images)
	np.random.set_state(state)
	np.random.shuffle(labels)

	validation_images = images[:5000]
	validation_labels = labels[:5000]
	train_images = images[5000:]
	train_labels = labels[5000:]

	return train_images,train_labels,validation_images,validation_labels








