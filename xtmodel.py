import numpy as np

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
tfpl = tfp.layers
tfpd = tfp.distributions

import xtdata

def build_vae_tfp(input_dim, latent_dim):
    prior = tfpd.Independent(tfpd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1)

    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[input_dim]),
        tfkl.Dense(units=200, activation=None, name='encoder_d1'),
        tfkl.LeakyReLU(),
        tfkl.Dense(units=150, activation=None, name='encoder_d2'),
        tfkl.LeakyReLU(),
        tfkl.Dense(units=120, activation=None, name='encoder_d3'),
        tfkl.LeakyReLU(),
        tfkl.Dense(units=100, activation=None, name='encoder_d4'),
        tfkl.LeakyReLU(),
        tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim), activation=None),
        tfpl.MultivariateNormalTriL(latent_dim, activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
    ])

    decoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[latent_dim]),
        tfkl.Dense(units=100, activation=None, name='decoder_d1'),
        tfkl.LeakyReLU(),
        tfkl.Dense(units=120, activation=None, name='decoder_d2'),
        tfkl.LeakyReLU(),
        tfkl.Dense(units=150, activation=None, name='decoder_d3'),
        tfkl.LeakyReLU(),
        tfkl.Dense(units=200, activation=None, name='decoder_d4'),
        tfkl.LeakyReLU(),
        tfkl.Dense(units=input_dim, activation=None, name='decoder_dout'),        
        tfpl.IndependentBernoulli([input_dim], tfpd.Bernoulli.logits),
    ])

    vae = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))

    return encoder, decoder, vae

class CustomCallback(tfk.callbacks.Callback):  
    def __init__(self, manager, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.manager = manager

    def on_epoch_end(self, epoch, logs=None):
        self.manager.save()

class ModelConfig: 
    def __init__(self, data, num_components=3, output_path='xtomics_data/ae_output'): 
        self.data = data
        self.num_components = num_components        
        self.output_path = output_path
        
        self.optimizer = tfk.optimizers.Adam(lr=0.001)  
    
        self.ds = self.data.tf_dataset()

        record = iter(self.ds.batch(1)).get_next()
        self.encoder, self.decoder, self.model = build_vae_tfp(input_dim=record.shape[1], latent_dim=self.num_components)
        
        negloglik = lambda x, rv_x: -rv_x.log_prob(x)
        self.model.compile(optimizer=self.optimizer, loss=negloglik)        

        self.checkpoint = tf.train.Checkpoint(model=self.model,
                                              encoder=self.encoder,
                                              decoder=self.decoder,
                                              opt=self.optimizer)
        
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.output_path, max_to_keep=3)

        status = self.checkpoint.restore(self.manager.latest_checkpoint)
    
    def train(self, batch_size, epochs):                
        print(self.model.summary())

        _ = self.model.fit(
            self.ds
              .map(xtdata.preprocess)
              .batch(batch_size), 
            epochs=epochs,
            callbacks=[CustomCallback(self.manager)])

    def embed(self, target_dim=3):
        embeds = []
        colors = []

        metadata = self.data.metadata()
        for i, (data_batch, labels_batch) in enumerate(self.data.h5_iter_with_labels(batch_size=5000, shuffle=False)):
            print(i)
            
            md_batch = metadata.loc[labels_batch]
        
            data,_ = xtdata.preprocess(data_batch)
            embed = self.encoder(data).sample(1)[0]

            embeds.append(embed)
            colors.append(md_batch['cluster_color'].values)

        embeds = np.concatenate(embeds)
        colors = np.concatenate(colors)

        if embeds.shape[1] > target_dim:
            print(f"model embedding is {embeds.shape[1]} dimensional, reducing to {target_dim}")
            
            import umap
            reducer = umap.UMAP(random_state=42, n_components=target_dim)
            embeds = reducer.fit_transform(embeds)

        return embeds, colors