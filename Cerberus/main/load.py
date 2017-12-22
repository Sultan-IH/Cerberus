import tensorflow as tf
from Cerberus.main import Network

"""to return an instance of Network"""


def load(name):
    """

    :param name: name of the model
    :return: network object, session
    """
    sess = tf.InteractiveSession()
    restored_model = tf.train.import_meta_graph('model_data/' + name + '.meta')
    restored_model.restore(sess, name)
    init_data = {
        "train_op": tf.get_collection('train_op')[-1],
        "compute_op": tf.get_collection('train_op')[-1],
        "placeholders": tf.get_collection('placeholders')[-1],
        "params": tf.get_collection('params')[0],
        "sess": sess
    }
    return Network(load_dict=init_data)
