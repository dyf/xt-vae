import tensorflow as tf
import pandas as pd
import h5py
import numpy as np

class XtomicsData:
    def __init__(self, 
        array_fname='array.h5', 
        array_ds='data', 
        labels_ds='sample_name',
        metadata_fname='metadata.csv'):

        self.array_fname = array_fname
        self.array_ds = array_ds
        self.labels_ds = labels_ds
        self.metadata_fname = metadata_fname

    def metadata(self):
        df = pd.read_csv(self.metadata_fname, index_col='sample_name')
        
        isna = pd.isna(df['cluster_color'])
        df.loc[isna,'cluster_color'] = '#CCCCCC'
        
        return df

    def h5_iter_with_labels(self, batch_size=None, shuffle=True):
        with h5py.File(self.array_fname, 'r') as hf:
            ds = hf[self.array_ds]
            lds = hf[self.labels_ds]

            idx = np.arange(ds.shape[0])
            if shuffle:
                np.random.shuffle(idx)

            if batch_size:
                for i in range(0, len(idx), batch_size):
                    batch_idx = idx[i:i+batch_size]
                    batch_idx.sort()

                    yield ds[batch_idx], lds[batch_idx]
            else:
                for i in idx:
                    yield ds[i], lds[i]

    def h5_iter(self, batch_size=None, shuffle=True):
        with h5py.File(self.array_fname, 'r') as hf:
            ds = hf[self.array_ds]

            idx = np.arange(ds.shape[0])
            if shuffle:
                np.random.shuffle(idx)

            if batch_size:
                for i in range(0, len(idx), batch_size):
                    batch_idx = idx[i:i+batch_size]
                    batch_idx.sort()

                    yield ds[batch_idx]
            else:
                for i in idx:
                    yield ds[i]

    def tf_dataset(self):
        with h5py.File(self.array_fname,'r') as f:
            shape = f[self.array_ds].shape
        
        return tf.data.Dataset.from_generator(
            lambda: self.h5_iter(),
            output_types=(tf.float32),
            output_shapes=tf.TensorShape([shape[1]]),
        )

def preprocess(x):
    x = tf.math.log(1+tf.cast(x, tf.float32))
    x = x / 10.0
    return x, x