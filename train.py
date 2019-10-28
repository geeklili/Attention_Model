# coding=utf-8
from keras.preprocessing import sequence
from keras.datasets import imdb
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import *
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# from Attention_keras import Attention, Position_Embedding
from numpy import float32
from sklearn.preprocessing import StandardScaler
from keras import regularizers
import keras


# attention模型
class Self_Attention(Layer):
	def __init__(self, output_dim, **kwargs):
		self.init = initializers.get('normal')
		self.supports_masking = True
		self.output_dim = output_dim
		super(Self_Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		# 为该层创建一个可训练的权重
		print('input_shape', input_shape)
		# inputs.shape = (batch_size, time_steps, seq_len)
		self.kernel = self.add_weight(name='kernel',
		                              shape=(3, input_shape[2], self.output_dim),
		                              initializer='uniform',
		                              trainable=True)

		super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

	def call(self, x):
		WQ = K.dot(x, self.kernel[0])
		WK = K.dot(x, self.kernel[1])
		WV = K.dot(x, self.kernel[2])

		print("WQ.shape", WQ.shape)

		print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

		QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

		QK = QK / (64 ** 0.5)

		QK = K.softmax(QK)

		print("QK.shape", QK.shape)

		V = K.batch_dot(QK, WV)

		return V

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1], self.output_dim)

	def get_config(self):
		config = {
			'output_dim': self.output_dim
		}
		base_config = super(Self_Attention, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

# 使用gpu
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# 加载数据
data = pd.read_csv('./data/train.csv')
data_x = data.iloc[:, 2:27]
data_y = data.iloc[:, 27]
data_y = pd.DataFrame([i if i < 10 else 10 for i in data_y.values.tolist()])
data_a = data.iloc[:, 28:]

# 可以根据数据直接训练
x = data_x.values
# a = data_a.values.reshape(-1, 3, 360)
a = data_a.values

# 也可以将数据进行标准化以后再训练数据
std_x = StandardScaler()
# std_a = StandardScaler()
# a = std_x.fit_transform(a)
# x = std_a.fit_transform(x)
a = a.reshape(-1, 3, 300)
y = pd.get_dummies(data_y.values.reshape(-1)).values

# 将数据分成训练集测试集
x_train, x_test, a_train, a_test, y_train, y_test = train_test_split(x, a, y, test_size=0.2)
x_train = np.array(x_train, dtype=float32)
x_test = np.array(x_test, dtype=float32)
a_train = np.array(a_train, dtype=float32)
a_test = np.array(a_test, dtype=float32)
y_train = np.array(y_train, dtype=float32)
y_test = np.array(y_test, dtype=float32)

# 输入数据1
inputA = Input(shape=(3, 300), dtype=float32)

# 同样可以使用LSTM进行数据的理解
# O_seq = LSTM(720, activation='relu', )(inputA)

O_seq = Self_Attention(360)(inputA)
O_seq = GlobalAveragePooling1D()(O_seq)

O_seq = BatchNormalization()(O_seq)
# O_seq = Dropout(0.5)(O_seq)
l_num = 0.001
O_seq = Dense(360, activation='relu', kernel_regularizer=regularizers.l2(l_num))(O_seq)
O_seq = BatchNormalization()(O_seq)

O_seq = Dense(360, activation='relu', kernel_regularizer=regularizers.l2(l_num))(O_seq)
O_seq = BatchNormalization()(O_seq)

O_seq = Dense(80, activation='relu', kernel_regularizer=regularizers.l2(l_num))(O_seq)
outputs_a = BatchNormalization()(O_seq)

model_a = Model(inputs=inputA, outputs=outputs_a)


# 输入数据2
inputB = Input(shape=(25,))
x = Dense(50, activation="relu", kernel_regularizer=regularizers.l2(l_num))(inputB)
x = BatchNormalization()(x)
# x = Dense(100, activation="relu", kernel_regularizer=regularizers.l2(l_num))(x)
# x = BatchNormalization()(x)
x = Dense(25, activation="relu", kernel_regularizer=regularizers.l2(l_num))(x)
x = BatchNormalization()(x)
# outputs_x = Dense(25, activation="relu")(x)
model_x = Model(inputs=inputB, outputs=x)

# 合并网络
combined = concatenate([model_a.output, model_x.output])

# 合并后的网络隐层
z = Dense(100, activation="relu", kernel_regularizer=regularizers.l2(l_num))(combined)
z = BatchNormalization()(z)
# z = Dense(200, activation="relu", kernel_regularizer=regularizers.l2(l_num))(z)
# z = BatchNormalization()(z)
z = Dense(100, activation="relu", kernel_regularizer=regularizers.l2(l_num))(z)
z = BatchNormalization()(z)
z = Dense(10, activation="softmax")(z)

# 合并后的模型输入输出
model_total = Model(inputs=[model_a.input, model_x.input], outputs=z)

print(model_total.summary())
# 尝试使用不同的优化器和不同的优化器配置
opt = Adam(lr=0.0002, decay=0.00001)
model_total.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print('Train...')
print(x_train)
h = model_total.fit([a_train, x_train], y_train, batch_size=100, epochs=50, validation_data=([a_test, x_test], y_test))

plt.plot(h.history["loss"], label="train_loss")
plt.plot(h.history["val_loss"], label="val_loss")
plt.plot(h.history["acc"], label="train_acc")
plt.plot(h.history["val_acc"], label="val_acc")
plt.legend()
plt.show()

# 模型的保存和加载
model_total.save("./data/position_predict.h5")
keras.models.load_model("./data/position_predict.h5", custom_objects={'Self_Attention': Self_Attention})
