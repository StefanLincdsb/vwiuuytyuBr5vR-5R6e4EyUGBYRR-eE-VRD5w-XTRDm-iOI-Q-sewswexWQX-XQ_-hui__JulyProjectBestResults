Train neural network on 4-class [OCT2017](https://data.mendeley.com/datasets/rscbjbr9sj/2) images to perform automatic classification, then run tests and statistical evaluations.

Images filenames, such as CNV-53018-1.jpeg, provide information about the subject id (53018), so because of suspicions of test set leakage (maybe high correlation within the same subject id?), a second train has been performed after removing from  train all images with ids that exist in test:

Full train:
classes: ['CNV', 'DME', 'DRUSEN', 'NORMAL']
train: 79484 images
val:   2000 images
calib: 2000 images
test:  1000 images
class weights: {CNV: 0.374, DME: 1.310, DRUSEN: 1.780, NORMAL: 0.536}

Train after image removal:
classes: ['CNV', 'DME', 'DRUSEN', 'NORMAL']
train: 49822 images
val:   2000 images
calib: 2000 images
test:  1000 images
class weights: {CNV: 0.318, DME: 1.102, DRUSEN: 2.288, NORMAL: 0.292}

29662 images were removed, hurting the training. Best test accuracy dropped from 99.9% to 98.2%.

PDF report available at [Proiect_Iulie_2026___ML_pentru_diagnostic.pdf](https://github.com/StefanLincdsb/vwiuuytyuBr5vR-5R6e4EyUGBYRR-eE-VRD5w-XTRDm-iOI-Q-sewswexWQX-XQ_-hui__JulyProjectBestResults/blob/main/Proiect_Iulie_2026___ML_pentru_diagnostic.pdf).

All scripts, models and results are available as release at [https://github.com/StefanLincdsb/vwiuuytyuBr5vR-5R6e4EyUGBYRR-eE-VRD5w-XTRDm-iOI-Q-sewswexWQX-XQ_-hui__JulyProjectBestResults/releases/tag/v2026_07_23](https://github.com/StefanLincdsb/vwiuuytyuBr5vR-5R6e4EyUGBYRR-eE-VRD5w-XTRDm-iOI-Q-sewswexWQX-XQ_-hui__JulyProjectBestResults/releases/tag/v2026_07_23).

Last edited scripts are in Python folder.

OCT2017 data downloaded from [https://data.mendeley.com/datasets/rscbjbr9sj/2](https://data.mendeley.com/datasets/rscbjbr9sj/2).

Direct download: [https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/5699a1d8-d1b6-45db-bb92-b61051445347/file_downloaded](https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/5699a1d8-d1b6-45db-bb92-b61051445347/file_downloaded).
