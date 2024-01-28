# Pretraining and Finetuning

This section involves advancing the capabilities of the Phi 2 language model through pretraining and finetuning stages, with a specific focus on vision-text alignment and instruction following.

## Overview

This phase of the project is bifurcated into two major stages: pretraining for vision-text alignment, and finetuning for instruction following. Both stages are crucial for enhancing the multimodal understanding of the model.

### Pretraining: Vision-Text Alignment

In the pretraining stage, we leverage a frozen pretrained Phi-2 language model and a frozen pretrained CLIP model, integrated with a custom projection layer. This setup is specifically designed for aligning visual and textual data.

#### Key Features
- Utilization of pretrained Phi-2 and CLIP models.
- Custom projection layer for aligning CLIP embeddings with Phi 2 embeddings.
- Training on a subset of the COCO 2017 dataset, encompassing 40,000 images and captions.

### Finetuning: Instruction Following

The finetuning stage is centered on enhancing the model's ability to follow instructions. This involves using the previously frozen CLIP model and the unfrozen pretrained Phi-2, along with the projection layer from the pretraining stage.

#### Key Features
- Frozen CLIP model and unfrozen pretrained Phi-2 for targeted training.
- Usage of a subset of the Instruct150K dataset with approximately 40,000 images.
- Exploration of Autoregressive Token Prediction with Teacher Forcing and Standard Language Model Training Method.

## Data Preparation

- The data preparation involved breaking down images into 49 patches using OpenAI CLIP ViT base 32, resulting in 7x7 patches.
- These patches were processed through the projection network to align the clip embeddings with the expected Phi 2 embeddings.
- The data included both the patch embeddings and the embeddings for a vision separation token, followed by text information (captions for pretraining, questions and answers for finetuning).

## Results and Analysis

[Summary of the results, performance metrics, and analysis from both pretraining and finetuning stages.]

## Challenges and Resolutions

### Handling Diverse Training Methods
- **Challenge**: Implementing and balancing two distinct training methods – Autoregressive Token Prediction with Teacher Forcing and the Standard Language Model Training Method – presented unique challenges, particularly in terms of computational efficiency and learning nuances.
- **Autoregressive Token Prediction with Teacher Forcing**: This method, while providing a controlled implementation of teacher forcing and potentially leading to better context consideration, faced issues with computational resource inefficiency. It limited batch sizes due to GPU memory constraints and required careful filtering of longer sequences.
- **Standard Language Model Training Method**: Although faster and more efficient, supporting larger batch sizes, and consistently applying teacher forcing, this method posed the risk of less nuanced learning, as it didn't adaptively use the model's predictions.
- **Resolution**: A balanced approach was adopted, where the benefits of both methods were leveraged. Autoregressive prediction was used in scenarios demanding deeper contextual understanding, while the standard method was applied in cases where efficiency and scale were prioritized.

### Dataset and Model Size Management
- **Challenge**: The sheer scale of datasets and model sizes posed significant challenges in terms of training time and resource allocation.
- **Resolution**: Techniques like gradient checkpointing and distributed training were employed to manage the large dataset sizes and model complexities efficiently. This enabled more effective training without compromising on the model's performance or the richness of the dataset.


## Future Directions

### Expanding Training Datasets
- The immediate plan for future work includes training the model on the full LLaVA Visual Instruct Pretrain LCS-558K dataset, which is a comprehensive subset of the LAION/CC/SBU dataset. This will provide a much broader range of visual-text data for pretraining, potentially enhancing the model's multimodal understanding.
- For the instruction fine-tuning stage, leveraging the full Instruct150K dataset will allow for more exhaustive training, further refining the model's instruction-following capabilities.

### Exploring Advanced CLIP Models and Projection Networks
- Another avenue for future exploration is the implementation of other CLIP models that utilize larger patches. This could lead to more detailed visual representations, contributing to improved model performance in visual-text alignment tasks.
- Additionally, developing a more complex projection network is on the horizon. Such a network would aim to achieve better visual representation and more effective visual-text alignment, potentially enhancing the overall efficacy of the model in processing and understanding multimodal data.

### Continuous Model and Training Improvement
- Continuous refinement of training methods, including exploring more efficient ways to implement autoregressive token prediction and standard language model training, will be a focus.
- Investigating other optimization techniques and model architectures to further enhance the performance and capabilities of the Phi 2 model in handling complex visual-text tasks remains a key area of ongoing research and development.


## References

- COCO 2017 Dataset: [Website](https://cocodataset.org/#home)
- Instruct150K Dataset: [Link to dataset or reference]
- OpenAI CLIP: [Learn More](https://openai.com/clip/)

