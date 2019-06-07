import numpy as np
np.random.seed(123)


from keras.layers import Layer
from keras.layers.wrappers import Wrapper
import keras.backend as K
from keras.engine import InputSpec
from keras import initializers, regularizers, constraints


import tensorflow as tf

	
class Attention(Layer):
	INPUT_NAMES = ["query", "key", "value"]
	def __init__(self, limit_context = -1, **kwargs):
		super(Attention, self).__init__(**kwargs)

		if not hasattr(limit_context, "__len__"):
			limit_context = (limit_context, limit_context)

		self.limit_context = limit_context

	def get_config(self):
		config = super(Attention, self).get_config()
		config["limit_context"] = self.limit_context
		return config

	def compute_output_shape(self, input_shape):
		input_shape = self._normalize_inputs(input_shape)

		if input_shape[0][-1] != input_shape[1][-1]:
			raise Exception("Queries and keys must have same final dimension")
		if input_shape[1][1] != input_shape[2][1]:
			raise Exception("Keys and values must have same timesteps")

		return input_shape[0][:-1] + (input_shape[-1][-1],)
	
	def _normalize_inputs(self, inputs):
		if not isinstance(inputs, list):
			inputs = [inputs]

		if len(inputs) < 3:
			for _ in range(3-len(inputs)):
				inputs.append(inputs[-1])

		if len(inputs) > 3:
			raise Exception("You cannot pass more than 3 inputs (query, key, value) to attention layer")

		return inputs

	def call(self, inputs, mask = None):
		inputs = self._normalize_inputs(inputs)
		
		queries = inputs[0]
		keys = inputs[1]
		values = inputs[2]
		
		matrix = K.sum(K.expand_dims(queries, 2) * K.expand_dims(keys, 1), axis = -1)
		# shape = (batchsize, queries, keys)

		matrix /= np.sqrt(K.int_shape(inputs[-1])[-1])

		if not mask is None:
			mask = K.cast(mask[2], K.floatx()) # (batchsize, values)
			mask = K.expand_dims(mask, 1)
			matrix *= mask # (batchsize, queries, values)
		
		length_q = K.shape(inputs[0])[1]
		length_v = K.shape(inputs[-1])[1]
		
		if self.limit_context[0] >= 0 or self.limit_context[1] >= 0:
			def _step(i):
				start = 0
				end = length_v

				if self.limit_context[0] >= 0:
					start = K.max([0, i-self.limit_context[0]])

				if self.limit_context[1] >= 0:
					end = K.min([length_v, i+self.limit_context[1]+1])
				
				context = values[:,start:end] # (batches, context, dim)
				attention = K.softmax(matrix[:,i,start:end], axis = -1) # (batches, context)
				
				return K.batch_dot(attention, context) # (batches, dim)

			tmp = K.map_fn(_step, K.arange(length_q), dtype = K.floatx()) # (queries, batches, dim)
			return K.permute_dimensions(tmp, [1,0] + list(range(2, K.ndim(tmp)))) # (batches, queries, dim)
			
		else:
			attention = K.softmax(matrix, axis=-1) # (batchsize, queries, keys)
			return K.batch_dot(attention, values)


	def compute_mask(self, inputs, mask = None):
		if not mask is None: 
			mask = self._normalize_inputs(mask)
			return mask[0]
		return mask

class TransformedAttention(Attention):

	def __init__(self, units,
		use_bias = True, 
		kernel_initializer = "glorot_uniform", 
		bias_initializer = "zeros", 
		kernel_constraint = None,
		bias_constraint = None,
		kernel_regularizer = None,
		bias_regularizer = None,
		**kwargs):
		super(TransformedAttention, self).__init__(**kwargs)

		if not hasattr(units, "__len__"): units = (units, units, units)
		if len(units) == 1: units = units * 3
		if len(units) == 2: units = (units[0],) + units
		self.units = units
		self.use_bias = use_bias
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		self.kernel_constraint = kernel_constraint
		self.bias_constraint = bias_constraint
		self.kernel_regularizer = kernel_regularizer
		self.bias_regularizer = bias_regularizer
	
	def get_config(self):
		config = super(TransformedAttention, self).get_config()
		config["units"] = self.units
		config["use_bias"] = self.use_bias
		config["kernel_initializer"] = self.kernel_initializer
		config["bias_initializer"] = self.bias_initializer
		config["kernel_constraint"] = self.kernel_constraint
		config["bias_constraint"] = self.bias_constraint
		config["kernel_regularizer"] = self.kernel_regularizer
		config["bias_regularizer"] = self.bias_regularizer
		return config

	def compute_output_shape(self, input_shape):
		input_shape = self._normalize_inputs(input_shape)
		att_shape = [shape[:-1] + (self.units[i],) for i, shape in enumerate(input_shape)]
		return super(TransformedAttention, self).compute_output_shape(att_shape)

	def build(self, input_shape):
		input_shape = self._normalize_inputs(input_shape)
		att_shape = [shape[:-1] + (self.units[i],) for i, shape in enumerate(input_shape)]
		super(TransformedAttention, self).build(att_shape)

		self.kernels = []
		
		if self.use_bias:
			self.biases = []

		for i, name in enumerate(self.INPUT_NAMES):
			self.kernels.append(self.add_weight(shape = (input_shape[i][-1], self.units[i]), 
				name = '{}_kernel_{}'.format(self.name, name),
				initializer = self.kernel_initializer,
				regularizer = self.kernel_regularizer,
				constraint = self.kernel_constraint))

			if self.use_bias:
				self.biases.append(self.add_weight(shape = (self.units[i],), 
					name = '{}_bias_{}'.format(self.name, name),
					initializer = self.bias_initializer,
					regularizer = self.bias_regularizer,
					constraint = self.bias_constraint))
		self.built = True

	
	def call(self, inputs, mask = None):
		inputs = self._normalize_inputs(inputs)
		if not mask is None:
			mask = self._normalize_inputs(mask)
		
		shapes = [K.shape(x) for x in inputs]

		transformed = [K.dot(x, kernel) for x, kernel in zip(inputs, self.kernels)]
		if self.use_bias:
			transformed = [x + bias for x, bias in zip(transformed, self.biases)]

		return super(TransformedAttention, self).call(transformed, mask = mask)
	
	
	
	
	
class MultiHeadAttention(Attention):
	
	def get_config(self):
		config = super(MultiHeadAttention, self).get_config()
		config["heads"] = self.heads
		config["units"] = self.units
		config["use_bias"] = self.use_bias
		config["kernel_initializer"] = self.kernel_initializer
		config["bias_initializer"] = self.bias_initializer
		config["kernel_constraint"] = self.kernel_constraint
		config["bias_constraint"] = self.bias_constraint
		config["kernel_regularizer"] = self.kernel_regularizer
		config["bias_regularizer"] = self.bias_regularizer
		return config

	def __init__(self, heads, 
		units = None, 
		use_bias = True, 
		kernel_initializer = "glorot_uniform", 
		bias_initializer = "zeros", 
		kernel_constraint = None,
		bias_constraint = None,
		kernel_regularizer = None,
		bias_regularizer = None,
		**kwargs):
		
		super(MultiHeadAttention, self).__init__(**kwargs)
		self.heads = int(heads)
		self.units = units

		if self.units:
			self.units = int(self.units)
		
		self.use_bias = use_bias
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)
		
	
	def compute_output_shape(self, input_shape):
		input_shape = self._normalize_inputs(input_shape)

		if self.units:
			input_shape = [shape[:-1] + (self.units,) for shape in input_shape]

		tmp = super(MultiHeadAttention, self).compute_output_shape(input_shape)
		return tmp[:-1] + (tmp[-1] * self.heads,)
		
	def build(self, input_shape):
		input_shape = self._normalize_inputs(input_shape)
		
		if self.units is None:
			self.units = input_shape[-1][-1]
		
		super(MultiHeadAttention, self).build([shape[:-1] + (self.units,) for shape in input_shape])

		self.kernels = []
		
		if self.use_bias:
			self.biases = []

		for i, name in enumerate(self.INPUT_NAMES):			
			self.kernels.append(self.add_weight(shape = (input_shape[i][-1], self.units * self.heads), 
				name = '{}_kernel_{}'.format(self.name, name),
				initializer = self.kernel_initializer,
				regularizer = self.kernel_regularizer,
				constraint = self.kernel_constraint))

			if self.use_bias:
				self.biases.append(self.add_weight(shape = (self.units * self.heads,), 
					name = '{}_bias_{}'.format(self.name, name),
					initializer = self.bias_initializer,
					regularizer = self.bias_regularizer,
					constraint = self.bias_constraint))


		self.built = True

	def call(self, inputs, mask = None):
		inputs = self._normalize_inputs(inputs)
		if not mask is None:
			mask = self._normalize_inputs(mask)
		
		shapes = [K.shape(x) for x in inputs]

		transformed = [K.dot(x, kernel) for x, kernel in zip(inputs, self.kernels)]
		if self.use_bias:
			transformed = [x + bias for x, bias in zip(transformed, self.biases)]

		stacks = [K.concatenate([x[:,:,i*self.units:(i+1)*self.units] for i in range(self.heads)], axis = 0) for x in transformed]
		
		if not mask is None:
			for i, m in enumerate(mask):
				if not m is None:
					mask[i] = K.concatenate([m for _ in range(self.heads)], axis = 0)
		
		output = super(MultiHeadAttention, self).call(stacks, mask = mask) # (heads * batchsize, timesteps, ...)

		output = K.reshape(output, (self.heads, shapes[0][0], shapes[0][1], -1)) # (heads, batchsize, timesteps, ...)

		output = K.concatenate([output[i] for i in range(self.heads)], axis = -1)
	
		return output	

class ParallelizedWrapper(Layer):
	def __init__(self, layers, devices, more = False, **kwargs):
		super(ParallelizedWrapper, self).__init__(**kwargs)

		self.devices = devices
		self.layers = layers
		self.more = more

		self.device_to_layers = {}
		for i, device in enumerate(self.devices):
			self.device_to_layers[device] = list(filter(lambda j:j%len(self.devices) == i, range(len(self.layers))))		

	def compute_output_shape(self, input_shape):
		return [layer.compute_output_shape(input_shape) for layer in self.layers]

	def compute_mask(self, inputs, mask = None):
		return [layer.compute_mask(inputs, mask) for layer in self.layers]

	def get_config(self):
		config = super(ParallelizedWrapper, self).get_config()
		config["devices"] = self.devices
		config["layers"] = [{"class_name": layer.__class__.__name__, "config": layer.get_config()} for layer in self.layers]
		
		return config

	def build(self, input_shape):
		for device in self.devices:
			with tf.device(device):
				for layernum in self.device_to_layers[device]:
					layer = self.layers[layernum]	
					if not layer.built:
						layer.build(input_shape)
		
		self.built = True

	def call(self, inputs, mask = None):
		output = [None for _ in range(len(self.layers))]
		for device in self.devices:
			with tf.device(device):
				for layernum in self.device_to_layers[device]:
					layer = self.layers[layernum]
					output[layernum] = layer.call(inputs, mask = mask)

		return output


def get_custom_objects():
	import sys, inspect
	
	return {name: obj for name, obj in inspect.getmembers(sys.modules[__name__], 
		lambda member: (inspect.isclass(member) or inspect.isfunction(member)) and member.__module__ == __name__)}

if __name__ == "__main__":
	import os
	import sys
	os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

	from keras.models import Model
	from keras.layers import Embedding, Input, GlobalMaxPooling1D, Dense, concatenate
	
	sys.path.append(".")
	from keras_layers import ParallelizedWrapper
	

	inp1 = Input((None, 2))
	inp2 = Input((None, 2))

	layers = [TransformedAttention(units = (5, 10)) for _ in range(2)]
	att = concatenate(ParallelizedWrapper(layers = layers, devices=["/gpu:0", "/gpu:1"])([inp1, inp2]))
	
	x = np.random.random(size=(3,6,2))
	y = np.random.random(size=(3,4,2))
	
	m = Model([inp1, inp2], [att])

	t = m.predict_on_batch([x,y])
	print(t)
	print(t.shape)
	
	from keras.datasets import imdb
	from keras.preprocessing.sequence import pad_sequences	

	vocab = 10000
	(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab, skip_top = 20)

	x_train = pad_sequences(x_train, maxlen = 100)
	x_test = pad_sequences(x_test, maxlen = 100)
	
	att = [ParallelizedWrapper(layers = [TransformedAttention(units=(10,50)) for _ in range(7)], devices = ["/gpu:0", "/gpu:1"]) for _ in range(5)]
	emb = Embedding(input_dim=vocab, output_dim = 200)

	
	inp = Input((None,))
	curr = emb(inp)
	for _ in range(3):
		curr = concatenate(att[_]([curr, curr, curr]))

	pool = GlobalMaxPooling1D()(curr)

	d = Dense(1, activation = "sigmoid")(pool)

	m = Model([inp], [d])
	m.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
	m.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

