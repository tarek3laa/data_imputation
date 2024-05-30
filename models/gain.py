import numpy as np
from utils.gain_utils import *
import tensorflow.compat.v1 as tf
from tqdm import tqdm


class Gain:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def fit_transform(self, data_x, hint_rate=0.9, alpha=100, iterations=int(1e4)):
        # Define mask matrix
        data_m = 1 - np.isnan(data_x)

        # Other parameters
        no, dim = data_x.shape

        # Hidden state dimensions
        h_dim = int(dim)

        # Normalization
        norm_data, norm_parameters = normalization(data_x)
        norm_data_x = np.nan_to_num(norm_data, 0)

        ## GAIN architecture
        # Input placeholders
        # Data vector
        X = tf.placeholder(tf.float32, shape=[None, dim])
        # Mask vector
        M = tf.placeholder(tf.float32, shape=[None, dim])
        # Hint vector
        H = tf.placeholder(tf.float32, shape=[None, dim])

        # Discriminator variables
        D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))  # Data + Hint as inputs
        D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

        D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
        D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

        D_W3 = tf.Variable(xavier_init([h_dim, dim]))
        D_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

        # Generator variables
        # Data + Mask as inputs (Random noise is in missing components)
        G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
        G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

        G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
        G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

        G_W3 = tf.Variable(xavier_init([h_dim, dim]))
        G_b3 = tf.Variable(tf.zeros(shape=[dim]))

        theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

        ## GAIN functions
        # Generator
        def generator(x, m):
            # Concatenate Mask and Data
            inputs = tf.concat(values=[x, m], axis=1)
            G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
            G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
            # MinMax normalized output
            G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
            return G_prob

        # Discriminator
        def discriminator(x, h):
            # Concatenate Data and Hint
            inputs = tf.concat(values=[x, h], axis=1)
            D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
            D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
            D_logit = tf.matmul(D_h2, D_W3) + D_b3
            D_prob = tf.nn.sigmoid(D_logit)
            return D_prob

        ## GAIN structure
        # Generator
        G_sample = generator(X, M)

        # Combine with observed data
        Hat_X = X * M + G_sample * (1 - M)

        # Discriminator
        D_prob = discriminator(Hat_X, H)

        ## GAIN loss
        D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                      + (1 - M) * tf.log(1. - D_prob + 1e-8))

        G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))

        MSE_loss = \
            tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

        D_loss = D_loss_temp
        G_loss = G_loss_temp + alpha * MSE_loss

        ## GAIN solver
        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

        ## Iterations
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Start Iterations
        for it in range(iterations):
            # Sample batch
            batch_idx = sample_batch_index(no, self.batch_size)
            X_mb = norm_data_x[batch_idx, :]
            M_mb = data_m[batch_idx, :]
            # Sample random vectors
            Z_mb = uniform_sampler(0, 0.01, self.batch_size, dim)
            # Sample hint vectors
            H_mb_temp = binary_sampler(hint_rate, self.batch_size, dim)
            H_mb = M_mb * H_mb_temp

            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

            _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                                      feed_dict={M: M_mb, X: X_mb, H: H_mb})
            _, G_loss_curr, MSE_loss_curr = \
                sess.run([G_solver, G_loss_temp, MSE_loss],
                         feed_dict={X: X_mb, M: M_mb, H: H_mb})

        ## Return imputed data
        Z_mb = uniform_sampler(0, 0.01, no, dim)
        M_mb = data_m
        X_mb = norm_data_x
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]

        imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

        # Renormalization
        imputed_data = renormalization(imputed_data, norm_parameters)

        return imputed_data
