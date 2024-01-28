# Phi 2 Language Model Training

This part of the project is dedicated to the training of the Phi 2 language model, exploring two distinct methodologies: Training from scratch and utilizing the LIT-GPT implementation.

## Overview

Training the Phi 2 model was undertaken with two different approaches to explore various aspects of language model training and to understand the nuances of each method.

### Training from Scratch

The Phi 2 model was developed from the ground up, featuring a custom architecture. This approach included an innovative addition of a mixture of experts after the MLP, aimed at enhancing the model's learning capabilities. The training utilized the diverse literary corpus of the Project Gutenberg dataset, providing a rich base for language understanding and development.

#### Key Features
- Custom-built architecture.
- Integration of a mixture of experts post-MLP.
- Training on Project Gutenberg dataset.

### LIT-GPT Implementation

In contrast to the from-scratch approach, the Phi 2 model was also trained using the LIT-GPT framework. This method leverages an established framework for training language models, focusing on the Redpajama dataset. The LIT-GPT implementation offered a different perspective on model training, emphasizing different dataset characteristics and training dynamics.

#### Key Features
- Utilization of the LIT-GPT framework.
- Training on the Redpajama dataset.
- Comparative study to the from-scratch training approach.

## Results

###################################################


## Challenges and Resolutions

### Model Scaling and GPU Utilization
- **Challenge**: The considerable size of the Phi 2 model posed significant challenges in model training, particularly regarding resource allocation and management.
- **Resolution**: Fully Sharded Data Parallel (FSDP) was implemented to efficiently split the model across multiple GPUs. This approach, coupled with gradient checkpointing, allowed for more effective management of computational resources, enabling the training of a large-scale model without compromising on performance.

### Training Efficiency
- **Challenge**: Balancing the computational intensity of training a large model with efficient use of resources.
- **Resolution**: By implementing techniques like gradient checkpointing, we were able to reduce the memory footprint, enabling us to train larger models or use larger batch sizes, thereby improving the overall training efficiency.


## Future Work

### Extending Training Duration and Implementing Optimization Techniques
- The immediate focus for future work involves extending the training duration of the Phi 2 model. Prolonged training periods are anticipated to significantly enhance the model's performance and capabilities.
- Incorporating advanced optimization techniques, such as flash attention, will be explored to further improve the training process and model efficiency.

### Enhancing Visual Instruction Fine-Tuning
- A key area of interest lies in the application of the Phi 2 model for visual instruction fine-tuning. Our unique setup, differing from Microsoft's approach, enables direct integration of language features into the language model. This is in contrast to Microsoft's Phi 2, which does not expose the cross-attention layer, limiting the direct incorporation of vision information.
- By training Phi 2 from scratch and designing its architecture, we have the opportunity to expose the cross-attention layer. This would allow for a more integrated approach, similar to how text features are incorporated in models like Stable Diffusion, enhancing the model's ability to process and understand multimodal inputs.
- This aspect of the project is particularly promising, considering the potential for more nuanced and effective integration of language and vision data, leading to more robust and versatile model applications.

### Addressing Open Source Limitations
- Another significant aspect of our work is addressing the limitations posed by Microsoft's Phi 2, which, while having open weights, does not have its model code released publicly.
- By developing our version of the Phi 2 model, we contribute to the open-source community, providing both the model code. This not only enhances transparency but also allows for broader usage and experimentation by the research community.


## References

- Lightning AI - Lit-GPT: [GitHub Repository](https://github.com/Lightning-AI/lit-gpt)
- Project Gutenberg: [Website](https://www.gutenberg.org/)
- Redpajama Dataset: [Link to dataset or reference]
- Phi 2: [Microsoft Phi 2](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models)
