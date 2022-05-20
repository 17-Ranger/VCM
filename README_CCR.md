train command example:(on 30901 with conda environment(ccr))
python train_vcm.py --config-file ./configs/Base-RCNN-FPN-VCM.yaml

with new training dataset or new server: first change ./vcm_coding/regis_vcmdata.py 
line176: root path

feature map in train process: ./detectron2/modeling/meta_arch/rcnn.py
line211: 'title_big' is the feature map after splice