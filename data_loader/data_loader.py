import os
import tensorflow as tf
from .data_augmentation import read_image

# Number of CPU cores, for parallelism
num_cpus = os.cpu_count()

class DataLoader(object):
    def __init__(self, filenames, all_filenames, batch_size, resolution, sess):
        self.filenames = filenames
        self.all_filenames = all_filenames
        self.batch_size = batch_size
        self.resolution = resolution
        self.num_cpus = num_cpus
        self.sess = sess
        
        self.data_iter_next = self.create_tfdata_iter(
            self.filenames, 
            self.all_filenames,
            self.batch_size,
            self.resolution
        )
        
    def create_tfdata_iter(self, filenames, fns_all_trn_data, batch_size, resolution):
        tf_fns = tf.constant(filenames, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(tf_fns)
        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=len(filenames)))
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                lambda filenames: tf.py_func(
                    func=read_image, 
                    inp=[filenames, 
                         fns_all_trn_data,
                         resolution], 
                    Tout=[tf.float32, tf.float32, tf.float32]
                ), 
                batch_size=batch_size,
                num_parallel_batches=self.num_cpus, # cpu cores
                drop_remainder=True
            )
        )
        dataset = dataset.prefetch(32)

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next() # this tensor can also be useed as Input(tensor=next_element)
        return next_element
        
    def get_next_batch(self):
        return self.sess.run(self.data_iter_next)
