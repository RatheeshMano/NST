import gradio as gr
import tensorflow as tf
import numpy as np

def process_and_style(content_image, style_image, style_weight=1e-2, content_weight=1e4, upscale_method="SRCNN", epochs=500):
    # Preprocess the input images
    content_image = tf.image.resize(content_image, (512, 512))
    style_image = tf.image.resize(style_image, (512, 512))
    content_image = tf.expand_dims(content_image, axis=0) * 255.0
    style_image = tf.expand_dims(style_image, axis=0) * 255.0

    # Extract features
    style_features, content_features = get_feature_representations(model, content_image, style_image)
    gram_style_features = [gram_matrix(feature) for feature in style_features]

    # Initialize target image
    init_image = tf.Variable(content_image, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=0.02)

    # Training loop
    for epoch in range(epochs):
        train_step(init_image, model, gram_style_features, content_features, optimizer)

    # Upscale the output image
    output_image = tf.squeeze(init_image, axis=0)
    if upscale_method == "SRCNN":
        output_image = tf.expand_dims(output_image, axis=0)
        output_image = srcnn.predict(output_image)[0]
    else:
        upscale_factor = 2
        new_height = output_image.shape[0] * upscale_factor
        new_width = output_image.shape[1] * upscale_factor
        output_image = tf.image.resize(output_image, (new_height, new_width), method='bicubic')

    output_image = tf.clip_by_value(output_image, 0, 255)
    output_image = tf.cast(output_image, tf.uint8)

    return output_image

# Gradio UI
content_image_input = gr.Image(label="Content Image", type="numpy")
style_image_input = gr.Image(label="Style Image", type="numpy")
style_weight_input = gr.Slider(1e-4, 1e0, value=1e-2, label="Style Weight")
content_weight_input = gr.Slider(1e3, 1e5, value=1e4, label="Content Weight")
epochs_input = gr.Slider(100, 20000, value=500, step=1000, label="Training Epochs")
upscale_method_input = gr.Radio(["SRCNN", "Bicubic"], value="SRCNN", label="Upscaling Method")

output_image_display = gr.Image(label="Output Image", type="numpy")

gr.Interface(
    fn=process_and_style,
    inputs=[
        content_image_input,
        style_image_input,
        style_weight_input,
        content_weight_input,
        upscale_method_input,
        epochs_input,
    ],
    outputs=output_image_display,
    title="Neural Style Transfer with SRCNN Upscaling",
    description="Upload content and style images to generate stylized outputs with upscaling."
).launch()
