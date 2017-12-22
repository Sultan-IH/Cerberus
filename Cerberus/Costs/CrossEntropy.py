import tensorflow as tf
from Cerberus.Costs.BaseCostClass import Cost


class CrossEntropy(Cost):
    def get_cost_fn(self, compute_op, labels, lam):
        print("cost_op")
        reg = self.l2_reg(lam)
        print(compute_op.get_shape())
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(compute_op, labels) + reg)

        return cost
