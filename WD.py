import matplotlib.pyplot as plt
import tensorflow as tf
import utils
import numpy as np
from sklearn.datasets import make_blobs

xs, ys = make_blobs(3000, centers=[[0, 0], [0, 10]], cluster_std=1)
xt, yt = make_blobs(3000, centers=[[50, -20], [50, -10]], cluster_std=1)

plt.scatter(xs[:, 0], xs[:, 1], c=ys, s=2, alpha=0.4)
plt.scatter(xt[:, 0], xt[:, 1], c=yt, s=2, cmap='cool', alpha=0.4)

l2_param = 1e-5
lr = 1e-3
num_step = 5000
batch_size = 64
tf.set_random_seed(0)

n_input = xs.shape[1]
num_class = 2
n_hidden = [20]

with tf.name_scope('input'):
    X = tf.placeholder(dtype=tf.float32)
    y_true = tf.placeholder(dtype=tf.int32)
    train_flag = tf.placeholder(dtype=tf.bool)
    y_true_one_hot = tf.one_hot(y_true, num_class)

with tf.name_scope('generator'):
    h1 = utils.fc_layer(X, n_input, n_hidden[0], layer_name='hidden1', input_type='dense')

with tf.name_scope('slice_data'):
    h1_s = tf.cond(train_flag, lambda: tf.slice(h1, [0, 0], [batch_size / 2, -1]), lambda: h1)
    h1_t = tf.cond(train_flag, lambda: tf.slice(h1, [batch_size / 2, 0], [batch_size / 2, -1]), lambda: h1)
    ys_true = tf.cond(train_flag, lambda: tf.slice(y_true_one_hot, [0, 0], [batch_size / 2, -1]), lambda: y_true_one_hot)

with tf.name_scope('classifier'):
    W_clf = tf.Variable(tf.truncated_normal([n_hidden[-1], num_class], stddev=1. / tf.sqrt(n_hidden[-1] / 2.)), name='clf_weight')
    b_clf = tf.Variable(tf.constant(0.1, shape=[num_class]), name='clf_bias')
    pred_logit = tf.matmul(h1_s, W_clf) + b_clf
    pred_softmax = tf.nn.softmax(pred_logit)
    clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit, labels=ys_true))
    clf_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys_true, 1), tf.argmax(pred_softmax, 1)), tf.float32))

alpha = tf.random_uniform(shape=[batch_size / 2, 1], minval=0., maxval=1.)
differences = h1_s - h1_t
interpolates = h1_t + (alpha*differences)
h1_whole = tf.concat([h1, interpolates], 0)


def critic(h):
    critic_h1 = utils.fc_layer(h, n_hidden[-1], 10, layer_name='critic_h1')
    out = utils.fc_layer(critic_h1, 10, 1, layer_name='critic_h2', act=tf.identity)
    return out

critic_out = critic(h1_whole)
critic_s = tf.cond(train_flag, lambda: tf.slice(critic_out, [0, 0], [batch_size / 2, -1]), lambda: critic_out)
critic_t = tf.cond(train_flag, lambda: tf.slice(critic_out, [batch_size / 2, 0], [batch_size / 2, -1]), lambda: critic_out)
wd_loss = (tf.reduce_mean(critic_s) - tf.reduce_mean(critic_t))
critic_interpolates = tf.cond(train_flag, lambda: tf.slice(critic_out, [batch_size, 0], [batch_size / 2, -1]), lambda: critic_out)
gradients = tf.gradients(critic_out, [h1_whole])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)

theta_C = [v for v in tf.global_variables() if 'classifier' in v.name]
theta_D = [v for v in tf.global_variables() if 'critic' in v.name]
theta_G = [v for v in tf.global_variables() if 'generator' in v.name]
wd_d_op = tf.train.AdamOptimizer(lr_wd_D).minimize(-wd_loss+gp_param*gradient_penalty, var_list=theta_D)
all_variables = tf.trainable_variables()
l2_loss = l2_param * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'weight' not in v.name])
total_loss = clf_loss + l2_loss + wd_param * wd_loss
train_op = tf.train.AdamOptimizer(lr).minimize(total_loss, var_list=theta_G + theta_C)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    S_batches = utils.batch_generator([xs, ys], batch_size / 2, shuffle=False)
    T_batches = utils.batch_generator([xt, yt], batch_size / 2, shuffle=False)

    for i in range(num_step):
        xs_batch, ys_batch = S_batches.next()
        xt_batch, yt_batch = T_batches.next()
        xb = np.vstack([xs_batch, xt_batch])
        yb = np.hstack([ys_batch, yt_batch])
        for _ in range(D_train_num):
            sess.run(wd_d_op, feed_dict={X: xb, y_true: yb, train_flag: True})
        _, l_wd = sess.run([train_op, wd_loss], feed_dict={X: xb, y_true: yb, train_flag: True})
        if i % 1 == 0:
            acc_xs, c_loss_xs = sess.run([clf_acc, clf_loss], feed_dict={X: xs, y_true: ys, train_flag: False})
            acc_xt, c_loss_xt = sess.run([clf_acc, clf_loss], feed_dict={X: xt, y_true: yt, train_flag: False})
            print 'step: ', i
            print 'wasserstein distance: %f' % l_wd
            print 'Source classifier loss: %f, Target classifier loss: %f' % (c_loss_xs, c_loss_xt)
            print 'Source label accuracy: %f, Target label accuracy: %f' % (acc_xs, acc_xt)
