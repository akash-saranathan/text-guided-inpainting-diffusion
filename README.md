# Advanced Image Inpainting via Text-Guided Segmentation and Stable Diffusion

## Introduction
This project implements a sophisticated pipeline for object-specific image inpainting. By combining *Grounding DINO* for zero-shot object detection and the *Segment Anything Model (SAM)* for precise mask generation, the system allows users to target specific image regions using natural language. The final inpainting is performed using a *Stable Diffusion* model, optimized via *LoRA (Low-Rank Adaptation)* to ensure high-fidelity, realistic reconstructions.

## Project Highlights
- **Precision Localization:** Integrated Grounding DINO with SAM to enable state-of-the-art, text-driven object segmentation.
- **Efficiency via LoRA:** Applied Low-Rank Adaptation fine-tuning to Stable Diffusion, significantly reducing trainable parameters while maintaining high output quality.
- **Performance Metrics:** Achieved substantial improvements in segmentation and reconstruction accuracy:
  - IoU: Improved from 0.8285 to 0.9734
  - Dice Score: Improved from 0.9062 to 0.9865
- **End-to-End Pipeline:** A seamless workflow from raw image + text prompt to a fully inpainted, realistic result.

## Skills & Tech Stack
- **Computer Vision:** Image Segmentation, Object Detection, Generative Modeling.
- **Deep Learning:** Stable Diffusion, LoRA Fine-Tuning, Transformer Architectures.
- **Frameworks:** PyTorch, Hugging Face Diffusers, Accelerate.
- **Natural Language Processing:** Text-to-Image alignment, Prompt Engineering.

## Dependencies
Key libraries and models used in this repository:
- torch & torchvision
- diffusers (Stable Diffusion)
- transformers (Grounding DINO)
- segment-anything (SAM)
- peft (LoRA implementation)

## Summary
The core of this research focuses on the intersection of language and vision. By leveraging the semantic understanding of Grounding DINO and the spatial precision of SAM, we create "smart masks" that guide the generative power of Stable Diffusion. The integration of LoRA fine-tuning demonstrates that high-quality inpainting can be achieved efficiently, making it viable for iterative research and deployment in resource-constrained environments.
