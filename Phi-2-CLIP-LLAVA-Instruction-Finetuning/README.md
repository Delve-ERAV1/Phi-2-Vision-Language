# Pretraining and Finetuning

This section involves advancing the capabilities of the Phi 2 language model through pretraining and finetuning stages, with a specific focus on vision-text alignment and instruction following.


This project's approach to pretraining and finetuning, particularly in the context of visual-text alignment and instruction following, aligns with the innovative strides made by [LLaVA](https://llava-vl.github.io/). The utilization of a subset of the COCO dataset for pretraining and the Instruct150K dataset for fine-tuning in your project mirrors LLaVA's use of comprehensive image-text datasets, demonstrating a similar commitment to advancing the multimodal capabilities of AI systems.

## Overview

This phase of the project is bifurcated into two major stages: pretraining for vision-text alignment, and finetuning for instruction following. Both stages are crucial for enhancing the multimodal understanding of the model.

### Pretraining: Vision-Text Alignment

In the pretraining stage, we leverage a frozen pretrained Phi-2 language model and a frozen pretrained CLIP model, integrated with a custom projection layer. This setup is specifically designed for aligning visual and textual data.

#### Key Features
- Utilization of pretrained Phi-2 and CLIP models.
- Custom projection layer for aligning CLIP embeddings with Phi 2 embeddings.
- Training on a subset of the COCO 2017 dataset, encompassing 40,000 images and captions.


## Sample Predictions
![image](https://github.com/Delve-ERAV1/Phi-2-Vision-Language/assets/11761529/8164799a-d6e8-4430-8c4c-891534fd3f85)
> A street sign in a big city street.

![image](https://github.com/Delve-ERAV1/Phi-2-Vision-Language/assets/11761529/d977021a-009c-4533-a940-97b4c5b0cd06)
> A man on a skateboard holds a dog on his arm.

![image](https://github.com/Delve-ERAV1/Phi-2-Vision-Language/assets/11761529/95b26d81-a5b2-4cf4-9e0e-5b4b1ae5a899)
> Street line along city with street lights and trees lit up by street lamps. 

![image](https://github.com/Delve-ERAV1/Phi-2-Vision-Language/assets/11761529/56cf9cfa-8381-4bf5-a65f-7a51a51f392e)
> A street filled with lots of people and tall buildings.  There are lots of trucks and lights.  There are also lots of pedestrians.



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

## wandb Logs

### Pretraining
[Standard Language Model Training Method](https://wandb.ai/sijpapi/MLM-Phi2-CLIP-Pretraining/reports/train-loss-24-01-28-15-23-25---Vmlldzo2NjYxMzgz?accessToken=i1hwm638s5rlw5emych7hz1h3wwpfjjxem4cpxxgjnoyiv0g3q19mjtbh5on0kwa)

[Autoregressive Token Prediction with Teacher Forcing](https://api.wandb.ai/links/sijpapi/yv6yoy5i)


```
predicted_captions- 60
target_captions  48
Image 2:
Target Caption: A man riding a motorcycle next to a busy street.
Predicted Caption: A man riding a motorcycle next to a busy street......
------------
predicted_captions- 81
target_captions  58
Image 3:
Target Caption: A dog stands next to a skateboard leaned up against a wall
Predicted Caption: A dog stands next to a skateboard leaning up against a wall wall wall wall
```


### Fine-Tuning
[Visual Instruction Following](https://wandb.ai/sijpapi/MLM-Phi2-CLIP-FINETUNING-Instruct150K/reports/loss-24-01-28-15-26-22---Vmlldzo2NjYxNDAw?accessToken=qoi4ty3r4vdclp18148rzgvf5sy0izv40ka3tq4jyikow8z344cwek74t9laah0w)

Sample 
```
Question: What is the cat doing in the image?
Actual Answer: The cat is laying down and watching a television program, with the light from the television screen casting on the cat.
Predicted Answer: The cat is sitting on the couch, looking at the TV screen, and staring at the image of the fish.
------------
Question: What is the person doing in the image?
Actual Answer: The person in the image is riding a skateboard down the side of a ramp,
Predicted Answer: The person in the image is skateboarding down a ramp, performing a trick by jumping off the ramp and landing on the skateboard ramp.
------------


## Challenges and Resolutions

### Handling Diverse Training Methods
- **Challenge**: Implementing and balancing two distinct training methods – Autoregressive Token Prediction with Teacher Forcing and the Standard Language Model Training Method – presented unique challenges, particularly in terms of computational efficiency and learning nuances.
- **Autoregressive Token Prediction with Teacher Forcing**: This method, while providing a controlled implementation of teacher forcing and potentially leading to better context consideration, faced issues with computational resource inefficiency. It limited batch sizes due to GPU memory constraints and required careful filtering of longer sequences.
- **Standard Language Model Training Method**: Although faster and more efficient, supporting larger batch sizes, and consistently applying teacher forcing, this method posed the risk of less nuanced learning, as it didn't adaptively use the model's predictions.
- **Resolution**: A balanced approach was adopted, where the benefits of both methods were leveraged. Autoregressive prediction was used in scenarios demanding deeper contextual understanding, while the standard method was applied in cases where efficiency and scale were prioritized.

### Dataset and Model Size Management
- **Challenge**: The sheer scale of datasets and model sizes posed significant challenges in terms of training time and resource allocation.
- **Resolution**: Techniques like gradient checkpointing and distributed training were employed to manage the large dataset sizes and model complexities efficiently. This enabled more effective training without compromising on the model's performance or the richness of the dataset.


### Projection Network with Mixture of Experts

```python

class PatchReducerWithProjections(nn.Module):
    def __init__(
        self,
        num_patches,
        reduced_num_patches,
        clip_embed,
        phi_embed,
        num_projection_layers=4,
        num_experts=6,
    ):
        super().__init__()

        self.moe = MixtureOfExperts(clip_embed, num_experts)
        self.output = nn.Linear(clip_embed, phi_embed)
        self.projection_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(clip_embed, clip_embed),
                    nn.GELU(),  
                    nn.Linear(clip_embed, clip_embed),
                )
                for _ in range(num_projection_layers)
            ]
        )

    def forward(self, x):
        
        x = x.permute(0, 2, 1) 
        x = self.patch_reducer(x)
        x = x.permute(0, 2, 1)
        x = self.moe(x)
        for layer in self.projection_layers:
            residual = x
            x = layer(x) + residual 
        x = self.output(x)
        return x
```


#### Teacher-Forcing

```python
def scheduled_sampling_rate(epoch, max_epochs, start_rate=1.0, end_rate=0.0, decay='exponential'):
    if decay == 'linear':
        return start_rate - (start_rate - end_rate) * (epoch / max_epochs)
    elif decay == 'exponential':
        return start_rate * (end_rate / start_rate) ** (epoch / max_epochs)
    elif decay == 'inverse_sigmoid':
        return end_rate + (start_rate - end_rate) * (max_epochs / (max_epochs + np.exp(epoch / max_epochs)))
```

```python
  sampling_rate = scheduled_sampling_rate(self.global_step, self.trainer.estimated_stepping_batches)
  if use_teacher_forcing and torch.rand(1).item() <= sampling_rate:
      next_token = target_captions[:, t-(num_patches+1)]
  else:
      next_token = torch.argmax(next_step_logits, dim=-1)
```


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


## Deployment

App is deployed to huggingface spaces. [Try it out here](https://huggingface.co/spaces/Sijuade/MLM-CLIP-PHI-2-LLAVA-chatbot)


## References

- COCO 2017 Dataset: [Website](https://cocodataset.org/#home)
- Instruct150K Dataset: [Link to dataset or reference]
- OpenAI CLIP: [Learn More](https://openai.com/clip/)

