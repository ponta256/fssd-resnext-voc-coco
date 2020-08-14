# FSSD-ResNeXT512

pytorch版SSDについて以下の改造を行ったFSSD-ResNeXT512のPascal VOCおよびCOCOの学習、評価、推論コード

* iterationベースからepochベースへの変更
* SSD512(入力サイズ512x512)のサポート
* warmup (burnin)の追加
* focal lossの追加
* prediction moduleの追加
* deconvolutionの追加
* FSSDの追加
* basenetをResNeXTに変更
* nmsを高速版に変更

## インストール

    $ git clone https://github.com/ponta256/fssd-resnext-voc-coco.git
    $ cd fssd-resnext-voc-coco
    $ mkdir weights

以下のファイルをダウンロードしてweighsの下に配置します。

[basenetのpretrainedウェイト](https://drive.google.com/open?id=1k0SXQbr4SR2-GYa0CXb-QRvBNFj9y7Ft)

[Pascal VOCで学習済みのウェイト (mAP=82.6@137epoch)](https://drive.google.com/open?id=1LILDY-tMxFSOOd3UdqrQHyQqs_ZO8ljp)

[COCOで学習済みのウェイト (AP=37.0@134epoch)](https://drive.google.com/open?id=1ErwmylDT286pjJEP2viy59ohO3voTK3V)

Pascal VOCとCOCOのデータを配置します。以下、配置例です(/mnt/ssd以下に配置した例)。Pascal VOCの方は入手したものを展開するだけなのでディレクトリだけ表示します。COCOの方は少し準備が必要なので以前の記事などをみてご用意ください。

    VOCdevkit/
    ├── VOC2007
    │   ├── Annotations
    │   ├── ImageSets
    │   │   ├── Layout
    │   │   ├── Main
    │   │   └── Segmentation
    │   ├── JPEGImages
    │   ├── SegmentationClass
    │   ├── SegmentationObject
    │   ├── annotations_cache
    │   └── results
    └── VOC2012
        ├── Annotations
        ├── ImageSets
        │   ├── Action
        │   ├── Layout
        │   ├── Main
        │   └── Segmentation
        ├── JPEGImages
        ├── SegmentationClass
        └── SegmentationObject

    coco/
    ├── annotations
    │   ├── captions_train2014.json
    │   ├── captions_val2014.json
    │   ├── instances_minival2014.json
    │   ├── instances_train2014.json
    │   ├── instances_trainval35k.json
    │   ├── instances_val2014.json
    │   ├── instances_valminusminival2014.json
    │   ├── person_keypoints_train2014.json
    │   └── person_keypoints_val2014.json
    └── images
        ├── minival2014 -> /mnt/ssd/coco/images/trainval35k
        ├── train2014
        │   ├── COCO_train2014_000000000009.jpg
        ...
        │   ├── COCO_train2014_000000581909.jpg
        │   └── COCO_train2014_000000581921.jpg
        ├── trainval35k
        │   ├── COCO_train2014_000000000009.jpg
        ...
        │   ├── COCO_val2014_000000581913.jpg
        │   └── COCO_val2014_000000581929.jpg
        └── val2014
            ├── COCO_val2014_000000000042.jpg
            ...
            ├── COCO_val2014_000000581913.jpg
            └── COCO_val2014_000000581929.jpg

### Pascal VOC学習

    $ python train_fssd_resnext.py --dataset=VOC --dataset_root=/mnt/ssd/VOCdevkit/ --batch_size=10 --weight_prefix=VOC512_FSSD_RESNEXT_

### COCO学習

    $ python train_fssd_resnext.py --dataset=COCO --dataset_root=/mnt/ssd/coco/ --batch_size=10 --weight_prefix=COCO512_FSSD_RESNEXT_

### Pascal VOC評価

    $ python eval_fssd_resnext_voc.py --trained_model=weights/VOC512_FSSD_RESNEXT_137.pth --voc_root=/mnt/ssd/VOCdevkit/
    
    AP for aeroplane = 0.8925
    AP for bicycle = 0.8743
    AP for bird = 0.8039
    AP for boat = 0.7574
    AP for bottle = 0.6961
    AP for bus = 0.8932
    AP for car = 0.8895
    AP for cat = 0.8867
    AP for chair = 0.6757
    AP for cow = 0.8649
    AP for diningtable = 0.7991
    AP for dog = 0.8800
    AP for horse = 0.8982
    AP for motorbike = 0.8840
    AP for person = 0.8488
    AP for pottedplant = 0.5945
    AP for sheep = 0.8726
    AP for sofa = 0.8122
    AP for train = 0.8818
    AP for tvmonitor = 0.8098
    Mean AP = 0.8258

### COCO評価

    $ python eval_fssd_resnext_coco.py --trained_model=weights/COCO512_FSSD_RESNEXT_134.pth --dataset_root=/mnt/ssd/coco/
    
    ~~~~ Mean and per-category AP @ IoU=[0.50,0.95] ~~~~
    37.0
    ~~~~ Summary metrics ~~~~
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.370
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.576
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.395
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.187
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.426
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.538
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.478
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.497
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.272
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.561
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.672

### Pascal VOC推論

    $ python pred_fssd_resnext.py --dataset=VOC --trained_model=weights/VOC512_FSSD_RESNEXT_137.pth /mnt/ssd/coco/images/val2014/COCO_val2014_000000123360.jpg

### COCO推論

    $ python pred_fssd_resnext.py --dataset=COCO --trained_model=weights/COCO512_FSSD_RESNEXT_134.pth /mnt/ssd/coco/images/val2014/COCO_val2014_000000123360.jpg
