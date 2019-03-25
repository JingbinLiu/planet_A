
import tensorflow as tf

import numpy as np

import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))  # <TensorSliceDataset shapes: (), types: tf.float64>

dataset2 = tf.data.Dataset.from_tensor_slices(
  (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.random.uniform(size=(5, 2)))
)             # <TensorSliceDataset shapes: ((), (2,)), types: (tf.float64, tf.float64)>

dataset3 = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    }
)   # <TensorSliceDataset shapes: {a: (), b: (2,)}, types: {a: tf.float64, b: tf.float64}>

a = [[1,2,3,4,5], [6,7,8,1,2], [10,1,2,3,2]]

a1 = tf.data.Dataset.from_tensors(a)


b = [[1,2,3,4,5], [6,7,8,1,2], [10,1,2,3,2]]

b1 = tf.data.Dataset.from_tensor_slices(b)

a2 = a1.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

b2 = b1.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
# == {[1,2,3,4,5,6,7,8,9,10]}


for one_element in tfe.Iterator(a1):
    print(one_element)
print('\n===============')
for one_element in tfe.Iterator(b1):
    print(one_element)
print('\n===============')
for one_element in tfe.Iterator(a2):
    print(one_element)
print('\n===============')
for one_element in tfe.Iterator(b2):
    print(one_element)
print('\n===============')
for one_element in tfe.Iterator(dataset2):
    print(one_element)


print('\n===============')
for one_element in tfe.Iterator(dataset2.batch(2)):
    print(one_element)

print('\n===============')
import itertools
tf.enable_eager_execution()

def gen():
  for i in itertools.count(1):
    yield (i, [[1] * i])

ds = tf.data.Dataset.from_generator(
    gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([1,None,])))

for value in ds.take(2):
  print (value)


print()