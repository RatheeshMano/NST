# Gradio Style Transfer Application

This project implements an image style transfer using TensorFlow and Gradio, hosted on **Heroku** and linked via **GitHub Pages** for easy access.

## Project Overview

The goal of this project is to demonstrate how to apply neural style transfer to images using pre-trained models, specifically VGG19, combined with a **Super-Resolution Convolutional Neural Network (SRCNN)** for upscaling. The app allows users to upload their own images and apply the style of a selected reference image. 

### Key Features:
- **Style Transfer**: Transform content images into a chosen artistic style.
- **Image Upscaling**: After style transfer, the image is upscaled using SRCNN for higher resolution.
- **Interactive Web Interface**: Built using **Gradio**, an easy-to-use interface that allows users to interact with the model via a web browser.

## Technologies Used
- **TensorFlow**: For implementing the neural network models, including VGG19 for style transfer.
- **Gradio**: For creating an easy-to-use web interface to interact with the models.
- **Heroku**: For hosting the application in the cloud.
- **GitHub Pages**: For linking to the hosted Gradio app.