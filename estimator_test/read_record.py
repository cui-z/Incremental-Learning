import tensorflow as tf

def map_func(example):
    # feature 的属性解析表
    feature_map = {'text': tf.FixedLenFeature([2], tf.float32),
                   'label': tf.FixedLenFeature([3], tf.float32)
                   }
    parsed_example = tf.parse_single_example(example, features=feature_map)

    # parsed_example["image"] 是 bytes 二进制数据，需要转化为 Tensor, 并告知转化后的 dtype
    # 我这里 np.random.rand 生成数据默认是 float64， 所以才这样写
    # 如果我们读取的图片数据是 0-255,则应该设置 out_type=tf.uint8
    text = parsed_example["text"]
    label = parsed_example["label"]
    return text, label

def file_based_input_fn_builder(input_file,batch_size):
    name_to_features = {
        "text": tf.FixedLenFeature([2], tf.float32),
        "label": tf.FixedLenFeature([3], tf.float32),
    }

    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        return example


        """The actual input function."""
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        # input_file 可以是个list 包含多个tf_record
    d = tf.data.TFRecordDataset(input_file)
    d = d.map(_decode_record).batch(4).repeat()
    return d




# dataset = tf.data.TFRecordDataset(["train.tf_record"])
# dataset = dataset.map(map_func=map_func)
# iterator = dataset.make_one_shot_iterator()
d  = file_based_input_fn_builder("train.tf_record",4)
# element = iterator.get_next()
next_op = d.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    for i in range(5):
        d = sess.run(next_op)
        print(d["text"])
        print(d["text"].shape)
        print("*")
        print(d["label"])
        print(d["label"].shape)
        print("---------------------")
