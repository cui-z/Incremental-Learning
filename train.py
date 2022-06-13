from __future__ import print_function
import tensorflow as tf
import numpy as np
import configparser
import re
import matplotlib.pyplot as mpl

trainning_task_file = 'train.cfg'
trainning_input_file = 'train_input.txt'
test_input_file = "test_set.txt"
model_path = './saved_model/'
# 读取配置
config = configparser.ConfigParser()
config.read(trainning_task_file, encoding='utf-8')
n = int(config['network']['input_nodes'])  # input vector size
K = int(config['network']['output_nodes'])  # output vector size
lam = float(config['network']['lambda'])
# 隐层规模 用逗号分开,类似 ”16,16,13“
hidden_layer_size = config['network']['hidden_layer_size']
# 分离字符
reobj = re.compile('[\s,\"]')
ls_array = reobj.split(hidden_layer_size);
ls_array = [item for item in filter(lambda x: x != '', ls_array)]  # 删空白
# 隐层个数
hidden_layer_elems = len(ls_array);

# 转为整形，并计入输出层
ns_array = []
for idx in range(0, hidden_layer_elems):
    ns_array.append(int(ls_array[idx]))
# Output is the last layer, append to last
ns_array.append(K)
# 总层数（含有输出层）
total_layer_size = len(ns_array)
# --------------------------------------------------------------
# create graph
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('network'):
        with tf.name_scope('input'):
            s = [n]
            a = [tf.placeholder(tf.float32, [None, s[0]], name="in")]
            W = []
            b = []
            z = []
            punish = tf.constant(0.0)
            for idx in range(0, total_layer_size):
                with tf.name_scope('layer' + str(idx + 1)):
                    s.append(int(ns_array[idx]))
                    W.append(tf.Variable(tf.random_uniform([s[idx], s[idx + 1]], 0, 1), name='W' + str(idx + 1)))
                    b.append(tf.Variable(tf.random_uniform([1], 0, 1), name='b' + str(idx + 1)))
                    z.append(tf.matmul(a[idx], W[idx]) + b[idx] * tf.ones([1, s[idx + 1]], name='z' + str(idx + 1)))
                    a.append(tf.nn.tanh(z[idx], name='a' + str(idx + 1)))
                with tf.name_scope('regular'):
                    punish = punish + tf.reduce_sum(W[idx] ** 2) * lam

    with tf.name_scope('loss'):
        y_ = tf.placeholder(tf.float32, [None, K], name="tr_out")
        loss = tf.reduce_mean(tf.square(a[total_layer_size]-y_),name="loss") + punish
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=a[total_layer_size]))
    with tf.name_scope("acc"):
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(a[total_layer_size], -1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # acc = tf.metrics.accuracy(labels=tf.argmax(y_, 1),
        #                           predictions=tf.argmax(a[total_layer_size], -1),
        #                           name='acc_op')
        #acc = tf.metrics.accuracy(tf.argmax(a[total_layer_size], -1), tf.argmax(y_, 1) ,name="my_acc")
    with tf.name_scope('trainning'):
        optimizer = tf.train.AdamOptimizer(name="opt")
        train = optimizer.minimize(loss, name="train")

    init = tf.global_variables_initializer()
    inint_loca = tf.local_variables_initializer()
    # save graph to Disk
    saver = tf.train.Saver()
# --------------------------------------------------------------
### create tensorflow structure end ###
sess = tf.Session(graph=graph)
sess.run(init)  # Very important
#sess.run(inint_loca) #tf.metrics.accuracy会产生两个局部变量



file_deal_times = int(config['performance']['file_deal_times'])
trunk = int(config['performance']['trunk'])
train_step = int(config['performance']['train_step'])
iterate_times = int(config['performance']['iterate_times'])
# trainning
x_data = np.zeros([trunk, n]).astype(np.float32)
# read n features and K outputs
y_data = np.zeros([trunk, K]).astype(np.float32)

total_red = 0

plot_x = []
plot_y = []


def get_test_data(test_file_path):
    with open(test_file_path,"rt") as f:
        lines= f.readlines()
        num_test = len(lines)
        x_test = np.zeros([num_test, n]).astype(np.float32)
        # read n features and K outputs
        y_test = np.zeros([num_test, K]).astype(np.float32)
        for lct in range(num_test):
            x_arr = reobj.split(lines[lct])
            x_arr = [item for item in filter(lambda x: x != '', x_arr)]  # remove null strings
            for idx in range(n):
                x_test[lct][idx] = float(x_arr[idx])
            for idx in range(K):
                y_test[lct][idx] = float(x_arr[idx + n])
        return x_test,y_test

x_test,y_test = get_test_data(test_input_file)

for rc in range(file_deal_times):
    with open(trainning_input_file, 'rt') as ftr:
        while 1:
            lines = ftr.readlines()
            if not lines:
                # reach end of file, run trainning for tail items if there is some.
                if (total_red > 0):
                    for step in range(iterate_times):
                        sess.run(train, feed_dict={a[0]: x_data[0:min(total_red, trunk) + 1],
                                                   y_: y_data[0:min(total_red, trunk) + 1]})
                break
            line_count = len(lines)
            for lct in range(line_count):
                x_arr = reobj.split(lines[lct])
                x_arr = [item for item in filter(lambda x: x != '', x_arr)]  # remove null strings
                for idx in range(n):
                    x_data[total_red % trunk, idx] = float(x_arr[idx])
                for idx in range(K):
                    y_data[total_red % trunk, idx] = float(x_arr[idx + n])
                total_red = total_red + 1
                # the trainning set run trainning
                if (total_red % train_step == 0):
                    # trainning
                    for step in range(iterate_times):
                        sess.run(train, feed_dict={a[0]: x_data[0:min(total_red, trunk) + 1],
                                                   y_: y_data[0:min(total_red, trunk) + 1]})
                    # print loss
                    lss,ac,bb = sess.run([loss,acc,b], feed_dict={a[0]: x_data[0:min(total_red, trunk) + 1],
                                                    y_: y_data[0:min(total_red, trunk) + 1]})

                    print("training rc:{0},total_red:{1},loss:{2},accuracy:{3}".format(rc,total_red, lss,ac))
                    # cop = sess.run(correct_prediction, feed_dict={a[0]: x_data[0:min(total_red, trunk) + 1],
                    #                                y_: y_data[0:min(total_red, trunk) + 1]})
                    # print(cop)
                    test_acc = sess.run(acc, feed_dict={a[0]: x_test,y_: y_test})
                    print("testing accuracy:{0}".format(test_acc))
                    print("b:{0}".format(bb[-1]))
                    plot_x.append(total_red)
                    plot_y.append(lss)
                    if (lss < 0.000001):
                        break

mpl.plot(plot_x, plot_y)
mpl.show()
# saving
# 保存，这次就可以成功了
saver.save(sess, model_path + '/model.ckpt')
