## Install instructions:

1. Download Bosch Small Traffic Lights Dataset to the 'data' folder
https://hci.iwr.uni-heidelberg.de/node/6132
2. Download pre-trained network and extract to the 'model' folder:
http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
3. Convert dataset into tensorflow record format with the following script (from /models/research/object_detection root folder):
`python traffic_lights/data/create_bosch_tf_record.py --label_map_path=traffic_lights/data/bosch_label_map.pbtxt --data_dir=traffic_ligths/data --output_dir=traffic_lights/data`

4. Train the model:
`python train.py --logtostderr --pipeline_config_path=traffic_lights/model/faster_rcnn_resnet101_bosch.config --train_dir=traffic_lights/model/train`

5. Evaluate the model:
`python eval.py --logtostderr --pipeline_config_path=traffic_lights/model/faster_rcnn_resnet101_bosch.config --checkpoint_dir=traffic_lights/model/train --eval_dir=traffic_lights/model/eval`
