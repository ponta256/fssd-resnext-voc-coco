# FSSD-ResNeXT512

pytorch版SSDについて以下の改造を行ったFSSD-ResNeXT512のPascal VOCおよびCOCOの学習、評価、推論コード

* iterationベースからepochベースへの変更
* SSD512(入力サイズ512x512)のサポート
* warmup (burnin)の追加
* focal lossの追加
* prediction moduleの追加
* deconvolutionの追加
* FSSDの追加

## インストール

    $ git clone https://github.com/ponta256/fssd-resnext-voc-coco.git
    $ cd fssd-resnext-voc-coco
    $ mkdir weights

以下のファイルをダウンロードしてweighsの下に配置します。

[basenetのpretrainedウェイト](https://drive.google.com/open?id=1k0SXQbr4SR2-GYa0CXb-QRvBNFj9y7Ft)

[Pascal VOCで学習済みのウェイト (137epoch)](https://drive.google.com/open?id=1LILDY-tMxFSOOd3UdqrQHyQqs_ZO8ljp)

[COCOで学習済みのウェイト (43epoch)](https://drive.google.com/open?id=1U6-QU4chjUiq_vrtpwN5RZPG7W5fDMBW)

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

### COCO VOC評価

    $ python eval_fssd_resnext_coco.py --trained_model=weights/COCO512_FSSD_RESNEXT_43.pth --dataset_root=/mnt/ssd/coco/

### Pascal VOC推論

    $ python pred_fssd_resnext.py --dataset=VOC --trained_model=weights/VOC512_FSSD_RESNEXT_137.pth /mnt/ssd/coco/images/val2014/COCO_val2014_000000123360.jpg

### COCO VOC推論

    $ python pred_fssd_resnext.py --dataset=COCO --trained_model=weights/COCO512_FSSD_RESNEXT_43.pth /mnt/ssd/coco/images/val2014/COCO_val2014_000000123360.jpg
