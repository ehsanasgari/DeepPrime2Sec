from keras.layers.merge import _Merge
from keras.layers import Layer, InputLayer

from keras import initializers
from keras import regularizers
from keras import constraints

import keras.backend as K

import tensorflow as tf

class NameLayer(Layer):
	def __init__(self, name, **kwargs):
		super(NameLayer, self).__init__(**kwargs)
		self.name = name
		self.supports_masking = True
	
	def get_config(self):
		config = super(NameLayer, self).get_config()
		config["name"] = self.name
		return config

class Print(Layer):
	def call(self, inputs, mask = None):
		if isinstance(inputs, list):			
			return [tf.Print(x, [x, K.shape(x)]) for x in inputs]
		return tf.Print(inputs, [inputs, K.shape(inputs)])

class LayerNorm(Layer):
	def __init__(self, scale_initializer='ones', bias_initializer='zeros', **kwargs):
		super(LayerNorm, self).__init__(**kwargs)
		self.epsilon = 1e-6
		self.scale_initializer = initializers.get(scale_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.supports_masking = True

	def build(self, input_shape):
		self.scale = self.add_weight(shape=(input_shape[-1],), 
		                             initializer=self.scale_initializer,
		                             trainable=True,
		                             name='{}_scale'.format(self.name))
		self.bias = self.add_weight(shape=(input_shape[-1],),
		                            initializer=self.bias_initializer,
		                            trainable=True,
		                            name='{}_bias'.format(self.name))
		self.built = True

	def call(self, x, mask = None):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		norm = (x - mean) * (1/(std + self.epsilon))
		return norm * self.scale + self.bias
	
	def get_config(self):
		config = super(LayerNorm, self).get_config()
		config["scale_initializer"] = self.scale_initializer
		config["bias_initializer"] = self.bias_initializer
		return config

class Unmask(Layer):
	def __init__(self, **kwargs):
		super(Unmask, self).__init__(**kwargs)
		self.supports_masking = True

	def compute_mask(self, inputs, mask = None):
		return None

class UnmaskValue(Unmask):
	def __init__(self, value = 0, **kwargs):
		super(Unmask, self).__init__(**kwargs)
		self.value = value

	def call(self, inputs, mask = None):
		if mask is None:
			return inputs

		for i in range(2, K.ndim(inputs)):
			mask = K.expand_dims(mask, -1)

		neg_mask = K.equal(mask, False)
		mask = K.cast(mask, K.floatx())
		neg_mask = K.cast(mask, K.floatx())
		
		return inputs * mask + neg_mask * self.value

	def get_config(self):
		config = super(UnmaskValue, self).get_config()
		config["value"] = self.value
		return config

class TransferMask(Layer):
	def compute_output_shape(self, input_shape):
		if not isinstance(input_shape, list) or len(input_shape) != 2:
			raise Exception("TransferMask layer takes exactly two inputs")

		if not input_shape[0][:len(input_shape[1])] == input_shape[1]:
			raise Exception("Input shapes do not match")
			
		return input_shape[1]

	def call(self, inputs, mask = None):
		return inputs[0]

	def compute_mask(self, inputs, mask = None):
		return mask[1]


class PositionEmbedding(Layer):
	def __init__(self, output_dim, mask_zero = False, **kwargs):
		super(PositionEmbedding, self).__init__(**kwargs)
		self.output_dim = output_dim
		self.mask_zero = mask_zero

	def call(self, inputs, mask=None):
		positions = K.zeros(K.shape(inputs)[:2]) + K.expand_dims(K.arange(K.shape(inputs)[1], dtype = K.dtype(inputs)), 0)
		
		outputs = []
		for i in range(self.output_dim):
			
			if i % 2: 
				func = K.sin
			else: 
				func = K.cos
			
			outputs.append(func(positions / 10000.0 ** (2*i/self.output_dim)))
				
		return K.stack(outputs, -1)

	def compute_output_shape(self, input_shape):
		return input_shape[:2] + (self.output_dim,)

	def compute_mask(self, inputs, mask=None):
		if not self.mask_zero:
			return None
		tmp = K.not_equal(inputs, 0)

		for _ in range(2, K.ndim(inputs)):
			tmp = K.any(tmp, axis = -1, keepdims = False)
		return tmp 
		
	def get_config(self):
		config = super(PositionEmbedding, self).get_config()
		config["mask_zero"] = self.mask_zero
		config["output_dim"] = self.output_dim
		return config


def reverse(x):
	return Reverser(1)(x)

class Reverser(Layer):
	def __init__(self, axis = 1, **kwargs):
		super(Reverser, self).__init__(**kwargs)
		self.axis = axis
	
	def _reverse(self, x):
		pattern = [self.axis] + [i if i != self.axis else 0 for i in range(1, K.ndim(x))]
		x = K.permute_dimensions(x, pattern)
		x = x[::-1]
		x = K.permute_dimensions(x, pattern)
		return x

	def call(self, inputs, mask = None):
		if isinstance(inputs, list):
			return [self._reverse(x) for x in inputs]
		return self._reverse(inputs)

	def compute_mask(self, inputs, mask = None):
		if mask is None:	
			return mask
		if isinstance(mask, list):
			return [self._reverse(m) for m in mask]
		return self._reverse(mask)
	
	def get_config(self):
		config = super(Reverser, self).get_config()
		config["axis"] = self.axis
		return config

class WeightedSum(_Merge):
	def __init__(self, with_scalar_weight = True, **kwargs):
		super(WeightedSum, self).__init__(**kwargs)
		self.with_scalar_weight = with_scalar_weight

	def build(self, input_shape):
		super(WeightedSum, self).build(input_shape)

		self.W = self.add_weight(shape = (len(input_shape),), initializer = 'random_uniform', name = '{}_W'.format(self.name))
		
		if self.with_scalar_weight:
			self.L = self.add_weight(shape = (1,), initializer = 'ones', name = '{}_L'.format(self.name))
	
		self.softmax = K.softmax(self.W)
	
		self.built = True

	def _merge_function(self, inputs):
		output = inputs[0] * self.softmax[0]
		for i in range(1, len(inputs)):
			output += inputs[i] * self.softmax[i]

		if self.with_scalar_weight:
			output *= K.sum(self.L)

		return output
	
	def get_config(self):
		config = super(WeightedSum, self).get_config()
		config["with_scalar_weight"] = self.with_scalar_weight
		return config

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

	def get_weights(self):
		return sum([layer.get_weights() for layer in self.layers], [])		

	@property
	def trainable_weights(self):
		if self.trainable:
			return sum([layer.trainable_weights for layer in self.layers if layer.trainable], [])	
		else:
			return []
	
	@property
	def non_trainable_weights(self):
		if not self.trainable:
			return sum([layer.non_trainable_weights for layer in self.layers], [])	
		else:
			return sum([layer.non_trainable_weights for layer in self.layers if not layer.trainable], [])	

	@property
	def updates(self):
		return sum([layer.updates for layer in self.layers if hasattr(layer, 'updates')], [])
	
	@property
	def losses(self):
		return sum([layer.losses for layer in self.layers if hasattr(layer, 'losses')], [])

	def compute_mask(self, inputs, mask = None):
		return [layer.compute_mask(inputs, mask) for layer in self.layers]

	@classmethod
	def from_config(cls, config, custom_objects=None):
		from keras.layers import deserialize as deserialize_layer
		layers = []
		for layer in config.pop("layers"):
			layers.append(deserialize_layer(layer, custom_objects=custom_objects))
		return cls(layers = layers, **config)

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

