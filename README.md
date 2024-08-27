# Drone Detection with LLM

## Overview
This repository hosts the code and resources for the Drone Detection project led by the research team at Universidad Europea de Madrid. The project explores the application of advanced models—QwenVL, PaliGemma, and Llava—integrated with vision capabilities to accurately detect drones in images.

## Introduction
The goal of this project is to develop a robust and efficient system for real-time drone detection using state-of-the-art large language models (LLMs) with visual understanding. By leveraging QwenVL, PaliGemma, and Llava, the system aims to deliver high accuracy and reliability in diverse and dynamic environments.

## Features
- **Real-time Drone Detection:** Swift and responsive detection capabilities.
- **High Accuracy and Precision:** Utilizes cutting-edge models to achieve superior accuracy.
- **Robust Environmental Adaptability:** Effective in various lighting and weather conditions.
- **Seamless Integration:** Easily integrates with existing surveillance or security systems.

## Dataset
The dataset comprises 400 images featuring drones in various environments. The task for each model is to determine the presence of a drone and, if detected, draw a rectangular bounding box around it. This detection and bounding box functionality is currently implemented using QwenVL, as PaliGemma and Llava focus on different aspects of image classification and do not generate bounding boxes.

## Models
This project leverages the following models for image classification:

- **QwenVL:** A high-accuracy model specifically designed for image recognition tasks, capable of detecting and highlighting drones within images.
- **PaliGemma:** Renowned for its robustness, this model excels in identifying objects under diverse environmental conditions but does not generate bounding boxes.
- **Llava:** Specializes in precise detection of small objects like drones, contributing to overall detection accuracy but without bounding box capabilities.

---

This version is focused on the key aspects of the project: the introduction, features, dataset, and models, making it concise and easy to follow.
