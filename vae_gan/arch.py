import numpy as np
import tensorflow as tf 
import os, random
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

SEED = 1234
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)
random.seed(SEED)

INPUT_DIM = (64,64,3)

CONV_FILTERS = [32,64,64, 128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

CONV_T_FILTERS = [64,64,32,3]
CONV_T_KERNEL_SIZES = [5,5,6,6]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

Z_DIM = 32

BATCH_SIZE = 100
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5

def sampling(args):
    z_mean, z_sigma = args
    epsilon = K.random_normal(shape=K.shape(z_sigma), mean=0.,stddev=1.)
    return z_mean + z_sigma * epsilon

def convert_to_sigma(z_log_var):
    return K.exp(z_log_var / 2)

class VAE_GAN():
    def __init__(self):
        self.models = self._build()
        self.full_model = self.models[0]
        self.encoder = self.models[1]
        self.encoder_mu_log_var = self.models[2]
        self.decoder = self.models[3]
        
        # Discriminators
        self.disc = self.models[4]
        self.gen_disc = self.models[5]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.learning_rate = LEARNING_RATE
        self.kl_tolerance = KL_TOLERANCE

    def _build(self):
        def vae_r_loss(y_true, y_pred):
            ######## y_true.shape = (batch size, 3, 64, 64)

            # y_true_flat = K.flatten(y_true)
            # y_pred_flat = K.flatten(y_pred)

            r_loss = K.sum(K.square(y_true - y_pred), axis = [1,2,3])
            return r_loss

        def vae_kl_loss(y_true, y_pred):

            kl_loss = - 0.5 * K.sum(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis = 1)
            kl_loss = K.maximum(kl_loss, KL_TOLERANCE * Z_DIM)
            return kl_loss

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)
        
        
        vae_x = Input(shape=INPUT_DIM, name='observation_input')
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0], name='conv_layer_1')(vae_x)
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0], name='conv_layer_2')(vae_c1)
        vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0], name='conv_layer_3')(vae_c2)
        vae_c4= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0], name='conv_layer_4')(vae_c3)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(Z_DIM, name='mu')(vae_z_in)
        vae_z_log_var = Dense(Z_DIM, name='log_var')(vae_z_in)
        vae_z_sigma = Lambda(convert_to_sigma, name='sigma')(vae_z_log_var)

        vae_z = Lambda(sampling, name='z')([vae_z_mean, vae_z_sigma])
        
        vae_z_input = Input(shape=(Z_DIM,), name='z_input')

        #### DECODER: we instantiate these layers separately so as to reuse them later
        vae_dense = Dense(1024, name='dense_layer')
        vae_z_out = Reshape((1,1,DENSE_SIZE), name='unflatten')
        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0], name='deconv_layer_1')
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1], name='deconv_layer_2')
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2], name='deconv_layer_3')
        vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3], name='deconv_layer_4')
        
        #### DECODER/GENERATOR IN FULL MODEL
        vae_dense_model = vae_dense(vae_z)
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1_model = vae_d1(vae_z_out_model)
        vae_d2_model = vae_d2(vae_d1_model)
        vae_d3_model = vae_d3(vae_d2_model)
        vae_d4_model = vae_d4(vae_d3_model)

        #### DECODER/GENERATOR ONLY
        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d4_decoder = vae_d4(vae_d3_decoder)
        
        #### DISCRIMINATOR
        disc_x = Input(shape=INPUT_DIM, name='disc_observation_input')
        disc_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0], name='disc_conv_layer_1')(disc_x)
        disc_c2 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0], name='disc_conv_layer_2')(disc_c1)
        disc_c3 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0], name='disc_conv_layer_3')(disc_c2)
        disc_c4 = Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0], name='disc_conv_layer_4')(disc_c3)
        disc_c5 = Flatten()(disc_c4)   
        disc_out = Dense(1, activation='sigmoid', name="discriminator_classification")(disc_c5)
        disc = Model(disc_x, disc_out)
        
        
        opti = Adam(lr=LEARNING_RATE)
        
        
        #### MODELS

        vae_full = Model(vae_x, vae_d4_model)
        vae_encoder = Model(vae_x, vae_z)
        vae_encoder_mu_log_var = Model(vae_x, (vae_z_mean, vae_z_log_var))
        vae_decoder = Model(vae_z_input, vae_d4_decoder)


        # Discriminator for fake data. 
        # For the combined model we will only train the generator/decoder
        vae_encoder.trainable = False
        disc.trainable = False
        gen_z = vae_encoder(vae_x)
        gen_x = vae_decoder(gen_z)
        gen_valid = disc(gen_x)
        gen_disc = Model(vae_x, gen_valid)

        disc.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy'])
        gen_disc.compile(optimizer=opti, loss='binary_crossentropy')
        vae_full.compile(optimizer=opti, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])
        
        return (vae_full,vae_encoder, vae_encoder_mu_log_var, vae_decoder, disc, gen_disc)

    
    def set_weights(self, filepath):
        # As now we are not using directly VAE so its no need to set weights. 
        #as weights are replaced in 2nd line with GAN weights.
        #self.full_model.load_weights(filepath)  
        self.gen_disc.load_weights(filepath)


    def train(self, data):
        # Shuffle the data
        n = data.shape[0]
        idxs = np.arange(0, n)
        np.random.shuffle(idxs)
        
        data_shuffled = data[idxs]
                          
        for i in range(0, n, BATCH_SIZE):
            data_batch = data_shuffled[i:(i+BATCH_SIZE)]
            
           
            # # Train the discriminator
            self.disc.trainable = True            

            # # Generate samples from VAE
            gen_batch = self.full_model.predict(data_batch)
            d_loss_real = self.disc.train_on_batch(data_batch, np.ones((BATCH_SIZE, 1)) )
            d_loss_fake = self.disc.train_on_batch(gen_batch, np.zeros((BATCH_SIZE, 1)) )
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)          


            # Train the generator
            self.disc.trainable = False
            self.encoder.trainable = True
            g_loss = self.gen_disc.train_on_batch(gen_batch, np.ones((BATCH_SIZE, 1)))
            

            # Train the VAE
            # self.disc.trainable = False            
            self.encoder.trainable = True
            vae_loss = self.full_model.train_on_batch(data_batch, data_batch)
            

            print('Disc. [loss, acc]={0}, Gen. loss[loss]={1}, VAE loss[loss, vae_r_loss, vae_kl_loss]={2}'.format(d_loss, g_loss, vae_loss))
            print()
        
    def save_weights(self, filepath):
        #There is no need to save weights of VAE. 
        #self.full_model.save_weights(filepath)
        self.gen_disc.save_weights(filepath)
