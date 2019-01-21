# AICamp1.Week9.Session4.Faster_RCNN

## Data path

Wrong data: zhouji1994/datasets/widerface/1

Fine tune check point: zhouji1994/datasets/widerface/1

Correct data: zhouji1994/datasets/correct-widerface-data/1

TF Object Detection code: martinhoest/datasets/tf_models/1

因为我们这里model用的参数，都是来自于其他已经train好的model里的参数，下面的文件都是包含已经train好的其他model里的参数

zhouji1994/datasets/widerface/1/checkpoint

zhouji1994/datasets/widerface/1/.DS_Store

之前train 好的model的参数都在这里：zhouji1994/datasets/widerface/1/model.ckpt.data-00000-of-00001

zhouji1994/datasets/widerface/1/pipeline.config

zhouji1994/datasets/widerface/1/model.ckpt.meta

zhouji1994/datasets/widerface/1/frozen_inference_graph.pb


## Run command

```
floyd login
```

### For TrainingFiles_1

```
cd TrainingFiles_1

floyd init '...'  # <project name>

floyd run --gpu --env tensorflow-1.5 --data zhouji1994/datasets/widerface/1:/training --data martinhoest/datasets/tf_models/1:/tf_models --tensorboard 'bash run.sh'
```

### For TrainingFiles_2

#### Training

```
cd TrainingFiles_2

floyd init '...' # <project name>

floyd run --gpu --env tensorflow-1.5 --data zhouji1994/datasets/widerface/1:/training --data martinhoest/datasets/tf_models/1:/tf_models --data zhouji1994/datasets/correct-widerface-data/1:/data --tensorboard 'bash run.sh'
```

#### Eval

```
floyd run --gpu --env tensorflow-1.5 --data zhouji1994/datasets/widerface/1:/training --data martinhoest/datasets/tf_models/1:/tf_models --data zhouji1994/projects/face-detection/27/output:/model --data zhouji1994/datasets/correct-widerface-data/1:/data --tensorboard 'bash run_eval.sh'
```
