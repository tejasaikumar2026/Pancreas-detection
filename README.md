This project focuses on pancreatic cancer detection using deep learning-based image segmentation.
It uses a U-Net model with a ResNet-18 encoder to detect tumor regions in CT scans and classify risk levels based on tumor size.

Features

Semantic segmentation using U-Net
Binary classification (Tumor / No Tumor)
Image-level prediction using segmentation output
ROC-AUC score calculation
Confusion matrix with performance metrics
Automatic threshold selection
Risk level classification:
No Tumor
Low Risk
Medium Risk
High Risk
Interactive visualization for predictions

🛠️ Installation

Install required dependencies:

pip install torch torchvision

pip install opencv-python

pip install albumentations

pip install segmentation-models-pytorch

pip install tqdm scikit-learn matplotlib

 Training

Run the training script:

python train.py --dataset_dir "your_dataset_path" --save_path "model_save_path"

Training Configuration

Image Size: 256 × 256

Batch Size: 8

Epochs: 15

Learning Rate: 1e-4

Loss Function

Dice Loss

Binary Cross Entropy (BCEWithLogitsLoss)

Combined Loss = 0.5 Dice + 0.5 BCE

 Evaluation Metrics

The model evaluates:

ROC Curve

AUC Score

Confusion Matrix

Accuracy

Sensitivity (Recall)

Specificity

Precision

F1 Score

 Testing & Risk Prediction

Run the testing script:

python test.py

 Risk Level Logic

Risk is calculated based on tumor size:

0 → No Tumor

Less than 0.2% → Low Risk

Less than 1% → Medium Risk

Greater than or equal to 1% → High Risk

🖼️ Visualization

Displays CT Scan and Predicted Mask side by side

Interactive viewer for selecting images

Shows predicted risk level

💾 Model Details

Architecture: Ensemble ResNet-18 and U-Net

Encoder: ResNet-18 (pretrained on ImageNet)

Decoder: U-Net

Input: CT Scan image

Output: Segmentation mask

 Results

Strong ROC-AUC performance

Balanced precision and recall

Effective detection of small tumor regions

 Key Highlights
 
 Combines segmentation + classification
 
 Uses medically relevant metrics

Provides risk-based interpretation

Optimized for moderate training time
 
 Future Improvements

EfficientNet / Transformer-based models

Multi-class tumor staging

3D CT scan support

Clinical validation
