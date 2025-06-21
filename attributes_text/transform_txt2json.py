import torch 
import json


def transform_txt2pt(cls_names,txt_path,filename, entries_per_cls=20):

    with open (txt_path,'r') as f:
        content = f.readlines()

    descriptors = {}
    index_up = 0
    for idx,cls in enumerate(cls_names):
        index_low = index_up + 2
        index_up  = index_low + entries_per_cls
        values = content[index_low:index_up]
        index_up +=2

        value = [f'a clean origami {cls}. ' + item.strip('  "').strip('",\n') for item in values]
        descriptors[cls] = value

    with open(filename, 'w') as fp:
        json.dump(descriptors, fp, indent=4)

    return True


if __name__ == "__main__":
    datasets = ['pascal_voc']

    voc_class_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
                   ]

    coco_class_list = ['person','bicycle','car','motorbike','aeroplane',
                    'bus','train','truck','boat','traffic light',
                    'fire hydrant','stop sign','parking meter','bench','bird',
                    'cat','dog','horse','sheep','cow',
                    'elephant','bear','zebra','giraffe','backpack',
                    'umbrella','handbag','tie','suitcase','frisbee',
                    'skis','snowboard','sports ball','kite','baseball bat',
                    'baseball glove','skateboard','surfboard','tennis racket','bottle',
                    'wine glass','cup','fork','knife','spoon',
                    'bowl','banana','apple','sandwich','orange',
                    'broccoli','carrot','hot dog','pizza','donut',
                    'cake','chair','sofa','pottedplant','bed',
                    'diningtable','toilet','tvmonitor','laptop','mouse',
                    'remote','keyboard','cell phone','microwave','oven',
                    'toaster','sink','refrigerator','book','clock',
                    'vase','scissors','teddy bear','hair drier','toothbrush']

    for data_name,cls_names in zip(datasets,[voc_class_list,coco_class_list]):
        txt_path = f'/reference_codes/CLIP/attributes_text/voc_cls_descriptions_it_3.txt'
        pt_path = f'/reference_codes/CLIP/attributes_text/descriptors_{data_name}_gpt4.0_cluster_a_photo_of4.json'
        entries_per_cls=20
        des_data = transform_txt2pt(cls_names,txt_path,pt_path,entries_per_cls)
