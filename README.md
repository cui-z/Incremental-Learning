# 增量学习的一个Demo

## 使用流程

- 使用create_train_file来创建训练样本
- 利用train来训练
- 使用Incre_train来增量学习

## 核心代码

- 保存模型

```python
saver.save(sess, model_path + '/model.ckpt')
```

- 恢复模型

```python
sess = tf.Session(graph=graph)
check_point_path = 'saved_model/'  # 保存好模型的文件路径
ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
saver.restore(sess, ckpt.model_checkpoint_path)
```
参考的这个文章[增量学习](https://blog.csdn.net/goldenhawking/article/details/78686415)

# 训练过程修改为高级API estimator版本

## 流程（根据Bert代码仿写）

- get_train_examples(train_file_path)  获取样本，输出结果为一个List，里面存的是一个类（包含feature和label）
- file_based_convert_examples_to_features(examples, output_file)  将List里面的数据写入到tf_record格式的文件中，利用tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))或者int生成相对应的feature，然后定义一个tf.train.Example，写入文件。
- file_based_input_fn_builder(input_file, batch_size, is_training)  利用tf.data.TFRecordDataset读取tf_record格式的文件，并且batch的生产出来，里面要注意：repeat函数只有在训练时使用
- my_model(features, labels, mode, params)  自己定义的模型，输入参数feature里面既可以包含feature也可以包含label,mode是用来判断被哪个方法调用。里面需要对train、eval、predict分别进行定义，my_model函数的返回值均是tf.estimator.EstimatorSpec，特别主要在eval的时候不能使用常规的准确率，可以使用tf.metrics函数（他是根据batch来计算最终的）。

## 使用方法

直接用estimator_test文件下的train.py即可。