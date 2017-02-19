import tensorflow as tf


class PoolLayer():
    dims = 4
    shape = None
    def get_op(self, x):
        """Should return its op"""
        self.op = self.max_pool_2x2(x)
        return self.op

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
