import tensorflow as tf


def save(sess, name):
    with sess:
        tf.train.export_meta_graph(filename='/model_data/' + name + '.meta')
