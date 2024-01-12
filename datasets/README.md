# Prepare datasets

If you have a dataset directory, you could use os environment variable named `YOLOX_DATADIR`. Under this directory, YOLOX and VOC will look for datasets in the structure described below, if needed.
```
$YOLOX_DATADIR/
  COCO/
  
or:

$YOLOX_DATADIR/
   VOCdevckit/
```
You can set the location for builtin datasets by
```shell
export YOLOX_DATADIR=/path/to/your/datasets
```
If `YOLOX_DATADIR` is not set, the default value of dataset directory is `./datasets` relative to your current working directory.

## Expected dataset structure for [COCO detection](https://cocodataset.org/#download):

```
COCO/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

## VOCDataset:
```
VOCdevkit/
    VOC2007/
        Annotations/
            0.xml, 1.xml, 2,xml, ...
        JPEGImages/
            0.jpg, 1.jpg, 2.jpg, ...
        ImagesSets/
            Main/
                test.txt  # 放置所有测试图像文件名
                trainval.txt   # 放置所有测试图像文件名，可以使用目录下label2voc.ipynb生成
```

You can use the 2014 version of the dataset as well.
