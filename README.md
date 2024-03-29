# Integrated Language and Vision Models

This project encompasses the development and integration of advanced language and vision models, demonstrating a comprehensive approach in machine learning. It involves training a custom language model, Phi 2, from scratch, and further enhancing it with vision-text alignment techniques for robust multimodal understanding.

## Overview

The project is divided into two main parts:

### Part 1: Phi 2 Language Model Training
Focusing on training the Phi 2 language model using two distinct methods - 

## Training from Scratch
In this approach, the Phi 2 language model is developed entirely from the ground up. A custom architecture was designed, including a novel addition of a mixture of experts after the MLP. This version of Phi 2 was trained on the comprehensive and diverse texts of the Project Gutenberg dataset, offering a unique opportunity to build and refine the language model's capabilities from a foundational level.

## LIT-GPT Implementation
This implementation involves training the Phi 2 model using the LIT-GPT framework, a well-established method for language model development. The training was conducted on the Redpajama dataset, providing a specific context and set of linguistic patterns for the model to learn. This approach served as a comparative study to the from-scratch training, offering insights into the model's performance under different training methodologies.

### Part 2: Pretraining and Finetuning
Pretraining and finetuning of the model for enhanced language and vision capabilities.

## 1: Vision-Text Alignment for Pretraining

In the pretraining stage, the project utilized a frozen pretrained Phi-2 language model and a frozen pretrained CLIP model, combined with a custom projection layer. This setup was specifically designed for vision-text alignment, training the projection layer to align CLIP embeddings with those expected by the Phi 2 model. The training employed a subset of 40,000 images and captions from the COCO 2017 dataset, focusing on developing the model's capability to understand and align visual and textual information.

## 2: Instruction Following Fine-Tunin

The fine-tuning stage aimed at enhancing the model's ability to follow instructions. This involved using a frozen CLIP model, an unfrozen pretrained Phi-2, and the unfrozen projection layer from the pretraining stage. The training utilized a subset of about 40,000 images from the Instruct150K dataset. This stage explored two key methods for dataset preparation: Autoregressive Token Prediction with Teacher Forcing and the Standard Language Model Training Method. These methods were instrumental in optimizing the model's performance for specific tasks such as instruction following, striking a balance between computational efficiency and the depth of learning.


## References

The following resources and papers have significantly contributed to the conceptualization and development of this project:

1. "Improved Baselines with Visual Instruction Tuning" - This paper provides insights into the fine-tuning processes for visual and language model integrations. [Read More](https://arxiv.org/abs/2310.03744)
2. "LLaVA: Large Language and Vision Assistant" - This research explores the integration of large language models with vision capabilities, offering foundational knowledge for this project. [Read More](https://arxiv.org/abs/2304.08485)
3. OpenAI CLIP - A cutting-edge model by OpenAI for learning visual concepts from natural language supervision. [Learn More](https://openai.com/clip/)
4. "Mixture of Experts Explained" - This Hugging Face blog post provides an overview of the Mixture of Experts architecture, a key concept in the development of the Phi 2 model. [Read More](https://huggingface.co/blog/moe)
5. Lightning AI - Lit-GPT - A GitHub repository offering an implementation framework for language models, used in part of this project. [Explore Repository](https://github.com/Lightning-AI/lit-gpt)
6. Rohan Shravan ERAv1 [The School Of AI](https://www.theschoolof.ai)
