# HCT-MFNet: A Hybrid Cross-Modal Feature Fusion Network for Lodged Wheat Segmentation

**HCT-MFNet** is a deep learning model designed for accurate segmentation of lodged wheat regions using multi-modal data, including RGB images and Digital Surface Models (DSM). The model leverages a hybrid feature fusion mechanism to effectively handle complex scenes and accurately identify the boundaries of lodged wheat regions.

## Key Features

- **Hybrid Cross-Modal Fusion**: HCT-MFNet integrates RGB and DSM data through advanced feature fusion techniques, such as the **Difference Complementation and Enhancement (DCE)** module and the **Dynamic Feature Fusion (DFF)** module. These mechanisms enhance the complementary features of the two modalities while suppressing redundant information.

- **Multimodal Fusion Transformer (MFT)**: At the core of the model is the **Multimodal Fusion Transformer (MFT)**, which improves the model's ability to capture global and local dependencies between modalities. The **Self-Cross Cooperative Attention (SCCA)** mechanism within the MFT helps to effectively fuse intra-modal and cross-modal information.

- **Attention Mechanisms**: The model uses both **channel-wise** and **spatial attention** to adaptively emphasize important features while suppressing irrelevant or noisy information, ensuring improved performance in complex background scenarios.

## Code Release
The complete code, along with pretrained models, will be publicly released after the paper is published. Stay tuned for further updates!
