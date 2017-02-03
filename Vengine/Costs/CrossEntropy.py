import tensorflow as tf
from Vengine.Costs.BaseCostClass import Cost


class CrossEntropy(Cost):

    def get_cost_fn(self, Y,labels, lam):
        print("cost_op")
        reg = self.l2_reg(lam)
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(Y, labels) + reg)

        return cost
