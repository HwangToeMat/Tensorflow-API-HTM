# Tensorflow-API-Master
## 0. Open-Images_EasyDownload 
#### <a href="https://github.com/HwangToeMat/Open-Images_EasyDownload">Helper library for downloading OpenImages categorically.</a>

## 1. Classification <a href="https://github.com/HwangToeMat/Tensorflow-API-HTM/blob/master/1.classification/reCAPTCHA_classification.ipynb">[Break through the reCAPTCHA]</a>
Break through the security program for prevent ing macros, reCAPTCHA , using pretrained model(<a href='http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'>Inception_Net</a>).
As you can see in the picture below, Inception_Net makes it easy to find a bus.
![image1](/1.classification/image/image0.jpg)

## 2. Object_detection <a href="https://github.com/HwangToeMat/Tensorflow-API-HTM/blob/master/2.object_detection/README.md">[Football play detection]</a>
We used **ssdlite_mobilenet_v2_coco<a href="http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz">[Download Link]</a>** to analyze soccer games in real time because we *need fast computing speed.* As you can see in the image below, ground truth(Right) recognizes people as one, but in our model(Left), we see one by one.
![result3](/2.object_detection/images/result3.png)
