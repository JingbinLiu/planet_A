
import tensorflow as tf

a = tf.Variable([2.0,],dtype=tf.float32)

b = tf.equal(a,2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
    print(sess.run(a))
    print(sess.run(tf.squeeze(a)))
