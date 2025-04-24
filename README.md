# SAR RARP 50 Challenge (2022)
Below are the details of the different aspects of implementation associated with the challenge. The UNet model was only run for 1 epoch due to hardware and time constraints.

## Dataset
The dataset is the [SAR-RARP50 dataset](https://www.synapse.org/#!Synapse:syn27618412/wiki/616881). The data comprises of 50 real-world RARP operations which have been recorded
with an endoscope and the data from the left endoscopic channel has been stored for each. Along with the video, labelled
data in the form of png files of the shape (1080, 1920) is provided. Each ’position’ of the labelled data corresponds to the
60th frame from the RARP video data which is available.
More information is present in the dataset, but it is not relevant for performing the surgical tool segmentation task. Each
value of the grayscale png files is a value between 0 and 9, inclusive, with 0 corresponding to the ’background’ class, and
1 corresponding to the ’tool clasper’. Information regarding the other classes can be found in the SAR-RARP50 Dataset’s
readme file.

## Training, Validation, and Test Datasets
The first 40 videos have been provided as the [training dataset](https://rdr.ucl.ac.uk/articles/dataset/SAR-RARP50_train_set/24932529), and the last 10 videos (video 41 to video 50) have been
provided as the [test dataset](https://rdr.ucl.ac.uk/articles/dataset/SAR-RARP50_test_set/24932499). The training dataset was further divided into the first 32 videos (operations) as the actual
training dataset, and the next 8 videos as the validation dataset for testing our model architecture for further improvements
before a final check on the testing dataset.
While splitting the training dataset, random frames (and their corresponding masks or labelled data) were not chosen across
different videos as that could result in frames from the same operation being present in both the training dataset, and the
validation dataset, leading the model to learn the styles and patterns of that particular surgery, causing artificially high
validation accuracy, while demonstrating poor generalization on new, unseen surgeries.

## Model Architecture
For the surgical tool segmentation task, we employed a UNet architecture implemented using the segmentation models.pytorch
(SMP) library. UNet is a popular encoder-decoder architecture widely used in biomedical image segmentation due to its ability
to capture both global context and fine-grained spatial details.
We used ResNet-34[3] as the encoder backbone, pre-trained on the ImageNet dataset. This encoder extracts multi-scale
hierarchical features from the input RGB frames, which are then passed through a symmetric decoder that progressively
upsamples the feature maps to produce dense pixel-wise predictions. Skip connections between corresponding encoder and
decoder layers preserve spatial information, improving segmentation accuracy, especially for small or fine structures.
The model was configured with:

• in channels = 3 (to match the RGB input).

• classes = 10 (to segment 10 distinct semantic categories in the surgical tool dataset).

For training, we used the Cross Entropy Loss function, which is suitable for multi-class segmentation problems. The
model was optimized using the Adam optimizer with a learning rate of 1 × 10−4
. Training was conducted for a single epoch
due to time and resource constraints, and model performance was evaluated using the mean Intersection over Union (mIoU)
metric on the validation set.

## Metrics and Evaluation
To assess the performance of the segmentation model, we conducted evaluations exclusively on the validation set after each
training epoch. Two key metrics were used:

• Cross Entropy Loss: This loss function was used during training and validation to quantify the discrepancy between
the predicted segmentation map and the ground truth labels. It is well-suited for multi-class pixel-wise classification
tasks and penalizes incorrect class predictions at the pixel level.

• Mean Intersection over Union (mIoU): This metric is widely adopted in semantic segmentation tasks. For each
class, the IoU is computed as the ratio of the intersection to the union of the predicted and ground truth regions. The
mIoU is then calculated as the mean across all segmentation classes. It provides an interpretable and class-sensitive
measure of segmentation quality.

## Results

| Training Condition  | Train Loss | Validation Loss | Validation mIoU |
| ------------- | ------------- | ------------- | ------------- |
| No Augmentations | 0.1591  | 0.1323 | 0.4663 |
| Horizontal Flip Augmentations | 0.6507 | 0.1967 | 0.3985 |
