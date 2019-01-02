class Weights_Tranined(object):
	def __init__(self, weights):
		self.weights = weights

		self.conv11_w = self.load_trained_weights('conv1_1_W')
		self.conv11_b = self.load_trained_weights('conv1_1_b')
		self.conv12_w = self.load_trained_weights('conv1_2_W')
		self.conv12_b = self.load_trained_weights('conv1_2_b')

		self.conv21_w = self.load_trained_weights('conv2_1_W')
		self.conv21_b = self.load_trained_weights('conv2_1_b')
		self.conv22_w = self.load_trained_weights('conv2_2_W')
		self.conv22_b = self.load_trained_weights('conv2_2_b')

		self.conv31_w = self.load_trained_weights('conv3_1_W')
		self.conv31_b = self.load_trained_weights('conv3_1_b')
		self.conv32_w = self.load_trained_weights('conv3_2_W')
		self.conv32_b = self.load_trained_weights('conv3_2_b')
		self.conv33_w = self.load_trained_weights('conv3_3_W')
		self.conv33_b = self.load_trained_weights('conv3_3_b')

		self.conv41_w = self.load_trained_weights('conv4_1_W')
		self.conv41_b = self.load_trained_weights('conv4_1_b')
		self.conv42_w = self.load_trained_weights('conv4_2_W')
		self.conv42_b = self.load_trained_weights('conv4_2_b')
		self.conv43_w = self.load_trained_weights('conv4_3_W')
		self.conv43_b = self.load_trained_weights('conv4_3_b')

		self.conv51_w = self.load_trained_weights('conv5_1_W')
		self.conv51_b = self.load_trained_weights('conv5_1_b')
		self.conv52_w = self.load_trained_weights('conv5_2_W')
		self.conv52_b = self.load_trained_weights('conv5_2_b')
		self.conv53_w = self.load_trained_weights('conv5_3_W')
		self.conv53_b = self.load_trained_weights('conv5_3_b')

		self.fc6_w = self.load_trained_weights('fc6_W')
		self.fc6_b = self.load_trained_weights('fc6_b')


		self.fc7_w = self.load_trained_weights('fc7_W')
		self.fc7_b = self.load_trained_weights('fc7_b')

		self.fc8_w = self.load_trained_weights('fc8_W')
		self.fc8_b = self.load_trained_weights('fc8_b')

	def load_trained_weights(self,name):
		return self.weights[name]

