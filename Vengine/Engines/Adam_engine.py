import tensorflow as tf
from Vengine.Engines.BaseEngine import Engine


class Adam_engine(Engine):
    def get_train_op(self, compute_op, labels):
        cost_fn = self.cost_class.get_cost_fn(compute_op, labels, 1e-2)
        train_step = tf.train.AdamOptimizer(self.lr).minimize(cost_fn)
        return train_step
