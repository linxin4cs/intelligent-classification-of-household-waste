# intelligent-classification-of-household-waste
English | [中文](./README.zh-CN.md) 

A repo to store my race code.(The 7th National College Students Engineering Training comprehensive ability Competition, China)

## Project Introduction
This project is a contestant work for the "7th National College Students' Engineering Training Comprehensive Ability Competition Selection" and is an intelligent waste classification system based on deep learning. The main work involves preprocessing the living waste dataset and pre-training a model using deep learning methods. It also involves using PyQt5 for interface design to realize the visualization of the intelligent waste classification system. Finally, it coordinates with a microcontroller to link the mechanical structure, achieving the automated operation of the intelligent waste classification system.

## Project Structure
The main structure of the project is as follows:
- recognition: Image recognition code
  - assets: Promotional videos
  - utils: Auxiliary functions
  - weights: Pre-trained models
  - main.py: Main program
  - UI.py: Interface design
- singlechip: Microcontroller code (machine code)
  - gpio_in.hex: Microcontroller input
  - gpio_out.hex: Microcontroller output
- train: Training code
  - data_sort.py: Process the raw data
  - folder_rename.py: Rename the folder from a number to the actual name
  - test.py: Test the model
  - train.py: Train the model
  - garbage_classify.json: As a mapping table for quick search and reference of each category's specific name

## Technology Stack
- PySerial: For serial communication
- OpenCV-Python: Provides tools for image and video processing, including capture, manipulation, and saving
- Pytorch: For computer vision
- threading: For executing concurrent operations
- PyQt5: For interface design
- ...