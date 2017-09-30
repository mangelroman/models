## Install Instructions

1. Download Bosch Small Traffic Lights Dataset to the 'data' folder
https://hci.iwr.uni-heidelberg.de/node/6132

2. Download pre-trained network and extract to the 'model' folder:
http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz

## Preparing the Dataset
The Tensorflow Object Detection API expects data to be in the TFRecord format, so we'll now run the create_bosch_tf_record script to convert from the raw Bosch Small Traffic Lights dataset into TFRecords. Run the following commands from the models/research/object_detection directory:

```bash
# From models/research/object_detection/ directory
python traffic_lights/data/create_bosch_tf_record.py \
    --label_map_path=traffic_lights/data/bosch_label_map.pbtxt \
    --data_dir=traffic_lights/data \
    --output_dir=traffic_lights/data
```

## Running the Training Job

A local training job can be run with the following command:

```bash
# From models/research/object_detection/ directory
python train.py --logtostderr \
    --pipeline_config_path=traffic_lights/model/faster_rcnn_resnet101_bosch.config \
    --train_dir=traffic_lights/model/train
```

By default, the training job will run indefinitely until the user kills it.

## Running the Evaluation Job

Evaluation is run as a separate job. The eval job will periodically poll the
train directory for new checkpoints and evaluate them on a test dataset. The
job can be run using the following command:

```bash
# From models/research/object_detection/ directory
python eval.py --logtostderr \
    --pipeline_config_path=traffic_lights/model/faster_rcnn_resnet101_bosch.config \
    --checkpoint_dir=traffic_lights/model/train \
    --eval_dir=traffic_lights/model/eval
```

As with the training job, the eval job will run indefinitely until terminated by default.

## Running Tensorboard

Progress for training and eval jobs can be inspected using Tensorboard. If
using the recommended directory structure, Tensorboard can be run using the
following command:

```bash
tensorboard --logdir=traffic_lights/model
```

Please note it may take Tensorboard a couple minutes to populate with data.

## Exporting a trained model for inference

After your model has been trained, you should export it to a Tensorflow
graph proto. A checkpoint will typically consist of three files:

* model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001,
* model.ckpt-${CHECKPOINT_NUMBER}.index
* model.ckpt-${CHECKPOINT_NUMBER}.meta

After you've identified a candidate checkpoint to export, run the following
command from models/research/object_detection:

``` bash
# From models/research/object_detection/ directory
python export_inference_graph.py --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH} \
    --output_directory output_inference_graph.pb
```

Afterwards, you should see a graph named output_inference_graph.pb.
