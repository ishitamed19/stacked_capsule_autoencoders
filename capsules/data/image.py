# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib import pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow import nest
import tensorflow_datasets as tfds

import os

import ipdb
st = ipdb.set_trace


from stacked_capsule_autoencoders.capsules.data import tfrecords as _tfrecords
from stacked_capsule_autoencoders.capsules.data.nlu import return_labels


def create(which,
					 batch_size,
					 subset=None,
					 n_replicas=1,
					 transforms=None,
					 **kwargs):
	"""Creates data loaders according to the dataset name `which`."""

	func = globals().get('_create_{}'.format(which), None)
	if func is None:
		raise ValueError('Dataset "{}" not supported. Only {} are'
										 ' supported.'.format(which, SUPPORTED_DATSETS))

	dataset = func(subset, batch_size, **kwargs)
	print('create dataaset: ', dataset)

	if transforms is not None:
		if not isinstance(transforms, dict):
			transforms = {'image': transforms}

		for k, v in transforms.items():
			transforms[k] = snt.Sequential(nest.flatten(v))

	def map_func(data):
		"""Replicates data if necessary."""
		data = dict(data)

		if n_replicas > 1:
			tile_by_batch = snt.TileByDim([0], [n_replicas])
			data = {k: tile_by_batch(v) for k, v in data.items()}

		if transforms is not None:
			img = data['image']

			for k, transform in transforms.items():
				data[k] = transform(img)

		return data


	def clevr_veggies_map_func(index, image, label):
		#st()
		data = {'index': index, 'image': image, 'label': label}

		if n_replicas > 1:
			print('n_replicas: ', n_replicas)
			tile_by_batch = snt.TileByDim([0], [n_replicas])
			data = {k: tile_by_batch(v) for k, v in data.items()}
			# print(data)

		if transforms is not None:
			img = data['image']
			# print('before transforms: ', data)

			for k, transform in transforms.items():
				data[k] = transform(img)
			# print('after transforms: ', data)

		return data



	if transforms is not None or n_replicas > 1:
		if 'clevr_veggies' in which:
			dataset = dataset.map(clevr_veggies_map_func) \
											 .prefetch(tf.data.experimental.AUTOTUNE)
		else:
			dataset = dataset.map(map_func)

	iter_data = dataset.make_one_shot_iterator()
	input_batch = iter_data.get_next()
	for _, v in input_batch.items():
		v.set_shape([batch_size * n_replicas] + v.shape[1:].as_list())

	return input_batch


def _create_mnist(subset, batch_size, **kwargs):
	return tfds.load(
			name='mnist', split=subset, **kwargs).repeat().batch(batch_size)
	

def _gen_clevr_veggies(directory,file_to_use, shuffle=False, first_only=False, rand_choice=False, do_argmax=False):
	text_file = open(os.path.join(directory, file_to_use), "r")
	img_list = text_file.readlines()
	count = len(img_list)

	img_dir = os.path.join(directory,'bd_l')
	#st()

	def _get_order():
		if shuffle:
			return np.random.choice(count, size=count, replace=False)
		else:
			return list(range(count))
	order = _get_order()

	i = 0
	while True:
		if order[i] < 10:
			f = '00000' + str(order[i]) + '.p'
		elif order[i] < 100:
			f = '0000' + str(order[i]) + '.p'
		else:
			f = '000' + str(order[i]) + '.p'
		data = pickle.load(open(os.path.join(img_dir, 'CLEVR_new_' + f),"rb"))

		# img = img.transpose(0, 2, 3, 1)

		if first_only:
			img = data['rgb_camXs_raw'][0][...,:3]
		elif rand_choice:
			img = data['rgb_camXs_raw'][random.choice(range(40))][...,:3]
		else:
			img = data['rgb_camXs_raw'][:][...,:3]


		if first_only or rand_choice:
			only_one = True
		else:
			only_one = False
		lbl = return_labels(data['tree_seq_filename'], only_one)
		
		# if do_argmax:
		#   lbl = np.argmax(lbl)
		yield order[i], img, lbl

		i += 1
		if i == count:
			order = _get_order()
			i = 0


def _create_clevr_veggies(subset, batch_size, first_only=True, rand_choice=False):
	#st() # imgs: [40, num_imgs, 3, 256, 256], lbsl: [num_imgs, 20] because multilabel.

	if subset=='train':
		file_to_use = 'bd_lt.txt'
	else:
		file_to_use = 'bd_lv.txt'
	g = lambda: _gen_clevr_veggies(
		os.path.join('/home/mprabhud/dataset/clevr_veggies/npys'),file_to_use,
		shuffle=subset == 'train',
		first_only=first_only,
		rand_choice=rand_choice
	)

	if first_only or rand_choice:
		output_shapes = ((), (256, 256, 3), (20))
	else:
		output_shapes = ((), (40, 256, 256, 3), (20))

	dataset = tf.data.Dataset.from_generator(
		g,
		output_types=(tf.int32, tf.int32, tf.int32),
		output_shapes=output_shapes
	)
	return dataset.repeat().batch(batch_size)



SUPPORTED_DATSETS = set(
		k.split('_', 2)[-1] for k in globals().keys() if k.startswith('_create'))
