import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = x1 * x2
print(result)
result = tf.multiply(x1,x2)
print(result)

a1 = tf.constant([[5]])
a2 = tf.constant([[6]])

#for matmul rank should be atleast 2 i.e the tensor should be 2 Dimensional
result = tf.matmul(a1, a2)
print(result)

session = tf.Session()
print(session.run(result))

session.close()

with tf.Session() as sess:
	print(sess.run(result))


