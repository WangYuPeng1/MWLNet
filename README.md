# Complementary-aware Synergistic Fusion : Advancing Precise Wheat Lodging Mapping Using UAV-Borne RGB Imagery and DSM

**MWLNet (Mapping Wheat Lodging Network)** is a deep learning model designed for accurate mapping lodged wheat using multi-modal data, including RGB images and Digital Surface Models (DSM). The model utilizes a hybrid feature fusion mechanism to effectively handle complex scenes and accurately identify the boundaries of lodged wheat.

## Key Features

- **Hybrid Cross-Modal Fusion**: MWLNet integrates RGB and DSM data through advanced feature fusion techniques, such as the **Discrepancy Modality Recalibration (DMR)** module and the **Dynamic Feature Consolidation (DFC)** module. These mechanisms enhance the complementary features of the two modalities while suppressing redundant information.

- **Bi-directional Semantic Synchronization (BSS)**: At the core of the model is the **Bi-directional Semantic Synchronization (BSS)**, which improves the model's ability to capture global and local dependencies between modalities. The **Self-Cross Cooperative Attention (SCCA)** mechanism within the BSS helps to effectively fuse intra-modal and cross-modal information.

- **Attention Mechanisms**: The model uses both **channel-wise** and **spatial attention** to adaptively emphasize important features while suppressing irrelevant or noisy information, ensuring improved performance in complex background scenarios.

## Code Release
The complete code, along with pretrained models, will be publicly released after the paper is published. Stay tuned for further updates!
