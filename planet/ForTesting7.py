
import tensorflow as tf

a = tf.Variable([2.0,],dtype=tf.float32)

b = tf.equal(a,2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    c = tf.constant([2.0, ], dtype=tf.float32)
    aa = a.assign([3.6,])
    with tf.control_dependencies([aa,]):
        bb = tf.equal(a,2)
    print(sess.run(b))
    print(sess.run(bb))
    print(sess.run(b))
    print(sess.run(a))
    print(sess.run(c))
    print(sess.run(tf.squeeze(a)))


print(a.graph)
print(c.graph)