import tensorflow as tf
import collections
import numpy as np
import os

class Input_example():
    def __init__(self, text, label):
        self.text = text
        self.label = label

def get_train_examples(train_file_path):
    input_result = []
    for line in open(train_file_path):
        line = line.strip().split(",")
        input_result.append(Input_example(text=[float(i) for i in line[:2]], label=[int(i) for i in line[2:]]))
    return input_result

def create_model(input_id, label):
    w1 = tf.Variable(tf.random_uniform([2, 10], 0, 1), name="w1")
    b1 = tf.Variable(tf.random_uniform([1], 0, 1)) ** tf.ones([1, 10])
    w2 = tf.Variable(tf.random_uniform([10, 15], 0, 1), name="w2")
    b2 = tf.Variable(tf.random_uniform([1], 0, 1)) ** tf.ones([1, 15])
    w3 = tf.Variable(tf.random_uniform([15, 12], 0, 1), name="w3")
    b3 = tf.Variable(tf.random_uniform([1], 0, 1)) ** tf.ones([1, 12])
    w4 = tf.Variable(tf.random_uniform([12, 3], 0, 1), name="w4")
    b4 = tf.Variable(tf.random_uniform([1], 0, 1)) ** tf.ones([1, 3])
    # b =  tf.Variable(tf.random_uniform([1], 0, 1), name='b')
    o1 = tf.nn.tanh(tf.matmul(input_id, w1) + b1)
    o2 = tf.nn.tanh(tf.matmul(o1, w2) + b2)
    o3 = tf.nn.tanh(tf.matmul(o2, w3) + b3)
    o4 = tf.nn.tanh(tf.matmul(o3, w4) + b4)
    return o4, w1, w2, w3, w4

# 返回 EstimatorSpec对象
def my_model(features, labels, mode, params):
    input_id = features["text"]
    label = features["label"]
    # 没好好改 流程跑通目前
    logits, w1, w2, w3, w4 = create_model(input_id, label)
    predicted_classes = tf.argmax(logits, -1, output_type=tf.int32)

    # loss = tf.losses.sparse_softmax_cross_entropy(labels=predicted_classes, logits=logits)+ \
    #            tf.reduce_sum(w1 ** 2) * params["lam"] + \
    #            tf.reduce_sum(w2 ** 2) * params["lam"] + \
    #            tf.reduce_sum(w3 ** 2) * params["lam"] + \
    #            tf.reduce_sum(w4 ** 2) * params["lam"]  效果差 不知道为啥
    loss = tf.reduce_mean(tf.square(logits - label), name="loss") + \
           tf.reduce_sum(w1 ** 2) * params["lam"] + \
           tf.reduce_sum(w2 ** 2) * params["lam"] + \
           tf.reduce_sum(w3 ** 2) * params["lam"] + \
           tf.reduce_sum(w4 ** 2) * params["lam"]

    # correct_prediction = tf.equal(tf.argmax(label, 1), predicted_classes)
    # acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 训练阶段返回的EstimatorSpec对象需要包含计算loss和最小化loss的操作（op）
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, name="train", global_step=tf.train.get_global_step())
        logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=1)
        output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,training_hooks=[logging_hook])
        # , training_hooks=[logging_hook])
    # 在评估阶段返回的EstimatorSpec对象需要包含计算metrics的操作，已经跟模型导出有个的操作
    elif mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=tf.argmax(label, 1),
                                       predictions=predicted_classes,
                                       name='acc_op')
        matrics = {"accuracy": accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=matrics)
    # 在预测阶段返回的EstimatorSpec对象需要包含跟获取预测结果有个的操作
    else:
        predictions = {
            "input_ids": input_id,
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    return output_spec

def file_based_convert_examples_to_features(examples, output_file):
    # 定义一个TFRecordWriter，用来写TFRecords文件
    writer = tf.python_io.TFRecordWriter(output_file)

    # 进行类型转换 接收参数是一个list
    # 有三种类型 tf.train.BytesList(value=[value])
    # tf.train.FloatList(value=[value])
    # tf.train.Int64List(value=[value])
    def create_float_feature(values):
        f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        return f

    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    for example in examples:
        # 用一个字典来存储
        features = collections.OrderedDict()
        features["text"] = create_float_feature(example.text)
        features["label"] = create_float_feature(example.label)

        # 定义一个example
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        # serializaToSreing  序列化  写入
        writer.write(tf_example.SerializeToString())
    writer.close()

def file_based_input_fn_builder(input_file, batch_size, is_training):
    # 定义一个特征词典，和写TFRecords时的特征词典相对应
    name_to_features = {
        "text": tf.FixedLenFeature([2], tf.float32),
        "label": tf.FixedLenFeature([3], tf.float32),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        # 根据上面的特征解析单个数据（保存时的单个Example）
        example = tf.parse_single_example(record, name_to_features)
        return example

    def input_fn(params):
        """The actual input function."""
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        # input_file 可以是个list 包含多个tf_record
        # 使用TFRecordDataset即可以作为tensorflow代码所写的模型的输入，也可以作为keras模型的输入
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size))
        """
        d = tf.data.TFRecordDataset(input_file)
        d = d.map(_decode_record).batch(4).repeat() 这样也可
        """
        # d = tf.data.TFRecordDataset(input_file)
        # d = d.map(_decode_record).batch(batch_size).repeat()
        return d

    return input_fn

def main():
    do_list = "train"
    train_file_path = "train_input.txt"
    output_file = "tmp/train.tf_record"
    is_training = False
    train_batch_size = 64
    eval_batch_size = 100
    num_train_epochs = 64
    params = {"lam": 0.000001}
    run_config = tf.estimator.RunConfig(model_dir='output_dir', save_checkpoints_steps=10, )

    # 传入的参数应该是模型超参数的一个集合，可以是一个dictionary。
    # 传入的配置用于指定模型如何运行训练和评估，以及在哪里存储结果。这个配置是一个RunConfig对象，该对象会把模型运行环境相关的信息告诉Estimator。
    # 模型函数是一个Python函数，它根据给定的输入构建模型
    estimator = tf.estimator.Estimator(model_fn=my_model, config=run_config, params=params)

    if do_list == "train":
        train_examples = get_train_examples(train_file_path)
        num_train_steps = int(len(train_examples) / train_batch_size * num_train_epochs)
        print("num_train_steps:{0}".format(num_train_steps))
        file_based_convert_examples_to_features(train_examples, output_file)
        train_input_fn = file_based_input_fn_builder(output_file, train_batch_size, is_training=True)
        print("----------------train---------------------")
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        do_list = "eval"

    if do_list == "eval":
        print("----------------eval---------------------")
        eval_file_path = "test_set.txt"
        eval_output_file = "tmp/eval.tf_record"
        eval_examples = get_train_examples(eval_file_path)
        file_based_convert_examples_to_features(eval_examples, eval_output_file)
        eval_input_fn = file_based_input_fn_builder(eval_output_file, eval_batch_size, is_training)
        result = estimator.evaluate(input_fn=eval_input_fn, )
        for key in sorted(result.keys()):
            print(" {0} = {1}".format(key, str(result[key])))
    if do_list == "predict":
        pass
        # result = estimator.predict(input_fn=eval_input_fn, )
        # for pre in result:
        #     print(pre["input_ids"])
        #     print(pre["probabilities"])
        #     print(pre["class_ids"])
        #     print("********")
        # print("----------------end---------------------")


if __name__ == '__main__':
    main()
