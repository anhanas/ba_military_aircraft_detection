"""
YOLOv26 OBB Military Aircraft Detection - Gradio Demo

A web interface for detecting military aircraft in satellite images using
YOLOv26 Oriented Bounding Box models.

Usage:
    python app.py

The app will start a local server and provide a shareable link.
"""

import time
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Model paths - all available YOLOv26 OBB models
MODEL_PATHS = {
    "YOLOv26 OBB Nano": "./weights/model-n.pt",
    "YOLOv26 OBB Small": "./weights/model-s.pt",
    "YOLOv26 OBB Medium": "./weights/model-m.pt",
    "YOLOv26 OBB Large": "./weights/model-l.pt",
    "YOLOv26 OBB X": "./weights/model-x.pt",
}

# Aircraft class names (20 classes)
CLASSES = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A10",
    "A11",
    "A12",
    "A13",
    "A14",
    "A15",
    "A16",
    "A17",
    "A18",
    "A19",
    "A20",
]

# Color palette for different classes (distinct colors)
COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),  # Maroon
    (0, 128, 0),  # Dark Green
    (0, 0, 128),  # Navy
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (64, 0, 0),  # Dark Red
    (0, 64, 0),  # Dark Green
    (0, 0, 64),  # Dark Blue
    (64, 64, 0),  # Dark Yellow
    (64, 0, 64),  # Dark Magenta
    (0, 64, 64),  # Dark Cyan
    (192, 0, 0),  # Bright Red
    (0, 192, 0),  # Bright Green
    (0, 0, 192),  # Bright Blue
]


def draw_obb(
    image: Image.Image,
    boxes: List[List[float]],
    confs: List[float],
    classes: List[int],
    draw_labels: bool = True,
    draw_filled: bool = False,
    alpha: float = 0.3,
) -> Image.Image:
    """
    Draw Oriented Bounding Boxes (OBB) on the image.

    Args:
        image: Input PIL Image
        boxes: List of OBB boxes in format [x_center, y_center, width, height, angle_in_degrees]
        confs: List of confidence scores for each box
        classes: List of class indices for each box
        draw_labels: Whether to draw class labels and confidence
        draw_filled: Whether to fill the OBB polygons with color
        alpha: Transparency for filled polygons (0-1)

    Returns:
        PIL Image with OBBs drawn
    """
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img_height, img_width = img.shape[:2]

    # Create an overlay for filled polygons
    overlay = img.copy() if draw_filled else None

    for box, conf, cls in zip(boxes, confs, classes):
        # OBB format: [x_center, y_center, width, height, angle]
        x_center, y_center, width, height, angle = box

        # Ensure coordinates are within image bounds
        x_center = max(0, min(img_width - 1, x_center))
        y_center = max(0, min(img_height - 1, y_center))

        # Calculate rotation
        angle_rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        half_w, half_h = width / 2, height / 2

        # Calculate corner points
        corners = np.array(
            [
                [
                    x_center + cos_a * half_w - sin_a * half_h,
                    y_center + sin_a * half_w + cos_a * half_h,
                ],
                [
                    x_center - cos_a * half_w - sin_a * half_h,
                    y_center - sin_a * half_w + cos_a * half_h,
                ],
                [
                    x_center - cos_a * half_w + sin_a * half_h,
                    y_center - sin_a * half_w - cos_a * half_h,
                ],
                [
                    x_center + cos_a * half_w + sin_a * half_h,
                    y_center + sin_a * half_w - cos_a * half_h,
                ],
            ]
        )

        corners = corners.astype(int)

        # Draw filled polygon on overlay
        if draw_filled and overlay is not None:
            cv2.fillPoly(overlay, [corners], COLORS[cls % len(COLORS)])

        # Draw polygon outline
        cv2.polylines(
            img, [corners], isClosed=True, color=COLORS[cls % len(COLORS)], thickness=2
        )

        # Draw center point
        cv2.circle(img, (int(x_center), int(y_center)), 3, (255, 255, 255), -1)

        # Draw label with class and confidence
        if draw_labels:
            label = f"{CLASSES[cls]}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_width, label_height = label_size

            # Draw label background
            label_x = max(0, corners[0][0])
            label_y = max(0, corners[0][1] - 5)
            cv2.rectangle(
                img,
                (label_x, label_y - label_height - 5),
                (label_x + label_width, label_y + 5),
                COLORS[cls % len(COLORS)],
                -1,
            )

            # Draw label text
            cv2.putText(
                img,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Blend overlay with original image if filled polygons were drawn
    if draw_filled and overlay is not None:
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return Image.fromarray(img)


def predict(
    image: Image.Image,
    model_name: str,
    conf_threshold: float = 0.5,
    draw_filled: bool = True,
    filter_classes: List[str] = [],
) -> Tuple[Image.Image, Dict[str, str], str]:
    """
    Perform prediction using the selected YOLOv26 OBB model.

    Args:
        image: Input image (PIL Image)
        model_name: Name of the model to use
        conf_threshold: Confidence threshold for detections
        draw_filled: Whether to fill OBB polygons
        filter_classes: List of class names to filter (empty = all classes)

    Returns:
        Tuple of (output_image, metrics, detection_summary)
    """
    try:
        # Load model (cache to avoid repeated loading)
        if not hasattr(predict, "model_cache"):
            predict.model_cache = {}

        if model_name not in predict.model_cache:
            model = YOLO(MODEL_PATHS[model_name])
            predict.model_cache[model_name] = model
        else:
            model = predict.model_cache[model_name]

        # Prepare image
        img = np.array(image)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Convert class names to indices for filtering
        filter_class_indices = []
        if filter_classes:
            for cls_name in filter_classes:
                if cls_name in CLASSES:
                    filter_class_indices.append(CLASSES.index(cls_name))

        # Perform inference
        start_time = time.time()
        results = model.predict(img, conf=conf_threshold, verbose=False)
        inference_time = time.time() - start_time
        fps = 1 / inference_time if inference_time > 0 else 0

        # Extract results (OBB format)
        boxes = []
        confs = []
        classes = []
        class_counts = {}

        for result in results:
            for box in result.obb:
                # YOLOv26 OBB uses xywhr format: [x_center, y_center, width, height, rotation]
                # Note: rotation is in radians, convert to degrees for draw_obb
                x_center, y_center, width, height, angle_rad = box.xywhr[0].tolist()
                angle_deg = np.rad2deg(angle_rad)
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                # Apply class filtering if specified
                if filter_class_indices and cls not in filter_class_indices:
                    continue

                boxes.append([x_center, y_center, width, height, angle_deg])
                confs.append(conf)
                classes.append(cls)

                # Count detections per class
                class_name = CLASSES[cls]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Draw OBBs on image
        output_img = draw_obb(
            image,
            boxes,
            confs,
            classes,
            draw_labels=True,
            draw_filled=draw_filled,
            alpha=0.3,
        )

        # Calculate metrics
        total_objects = len(boxes)
        avg_confidence = np.mean(confs) if confs else 0

        metrics = {
            "Model": model_name,
            "Inference Time (s)": f"{inference_time:.4f}",
            "FPS": f"{fps:.2f}",
            "Total Objects Detected": total_objects,
            "Average Confidence": f"{avg_confidence:.3f}",
            "Confidence Threshold": f"{conf_threshold:.2f}",
            "Classes Detected": class_counts if class_counts else {},
        }

        # Create detection summary
        if total_objects == 0:
            summary = "No aircraft detected."
        else:
            summary_lines = [
                f"Detected {total_objects} aircraft with {model_name}",
                f"Average confidence: {avg_confidence:.2%}",
                f"Inference time: {inference_time:.3f}s ({fps:.1f} FPS)",
            ]
            if class_counts:
                summary_lines.append("Class distribution:")
                for cls_name, count in sorted(class_counts.items()):
                    summary_lines.append(f"  {cls_name}: {count}")
            summary = "\n".join(summary_lines)

        return output_img, metrics, summary

    except Exception as e:
        # Return error state
        error_msg = f"Error during prediction: {str(e)}"
        metrics = {"Error": error_msg}
        summary = error_msg
        return image, metrics, summary


def get_model_info(model_name: str) -> str:
    """Get information about the selected model."""
    model_sizes = {
        "YOLOv26 OBB Nano": "~6MB - Fastest, lowest accuracy",
        "YOLOv26 OBB Small": "~21MB - Balanced speed/accuracy",
        "YOLOv26 OBB Medium": "~48MB - Good balance",
        "YOLOv26 OBB Large": "~57MB - High accuracy",
        "YOLOv26 OBB X": "~126MB - Largest, highest accuracy",
    }
    return model_sizes.get(model_name, "Unknown model")


# Create Gradio interface
with gr.Blocks(title="YOLOv26 OBB: Military Aircraft Detection") as demo:
    gr.Markdown(
        """
        # **YOLOv26 OBB: Military Aircraft Detection on Satellite Images**

        Detect and classify **military aircraft** in satellite imagery using **YOLOv26 Oriented Bounding Box** models.
        These models are specifically trained for detecting 20 different aircraft classes in overhead imagery.

        **Features:**
        - Oriented Bounding Box (OBB) detection for precise aircraft localization
        - 20 different aircraft class classifications
        - Multiple model sizes (Nano to X) for speed/accuracy trade-offs
        - Adjustable confidence threshold
        - Fill option for better visualization
        - Detailed performance metrics
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")

            # Image upload
            input_image = gr.Image(
                label="Upload Satellite Image",
                type="pil",
                height=400,
            )

            # Model selection
            model_selector = gr.Dropdown(
                choices=list(MODEL_PATHS.keys()),
                value="YOLOv26 OBB Nano",
                label="Select YOLOv26 OBB Model",
            )

            # Model info display
            model_info = gr.Textbox(
                label="Model Info",
                value=get_model_info("YOLOv26 OBB Nano"),
                interactive=False,
                lines=1,
            )

            # Update model info when model changes
            model_selector.change(
                fn=get_model_info,
                inputs=[model_selector],
                outputs=[model_info],
            )

            # Confidence threshold slider
            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.99,
                value=0.5,
                step=0.05,
                label="Confidence Threshold",
                info="Only show detections with confidence above this value",
            )

            # Fill polygons checkbox
            fill_checkbox = gr.Checkbox(
                value=True,
                label="Fill OBB Polygons",
                info="Fill bounding boxes with color for better visibility",
            )

            # Class filter (multi-select)
            class_filter = gr.CheckboxGroup(
                choices=CLASSES,
                label="Filter Classes (Optional)",
                info="Select specific classes to detect (empty = all classes)",
            )

            # Predict button
            predict_btn = gr.Button("Detect Aircraft", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Output")

            # Output image
            output_image = gr.Image(
                label="Detection Results",
                height=400,
                interactive=False,
            )

            # Detection summary
            summary_output = gr.Textbox(
                label="Detection Summary",
                lines=8,
                interactive=False,
            )

            # Performance metrics
            metrics_output = gr.JSON(
                label="Performance Metrics",
                height=200,
            )

    # Example images
    gr.Markdown("### Try with Example Images")
    gr.Examples(
        examples=[
            ["examples/example1.jpg", "YOLOv26 OBB Medium", 0.5, True, []],
            ["examples/example2.jpg", "YOLOv26 OBB X", 0.4, True, []],
            ["examples/example3.jpg", "YOLOv26 OBB Large", 0.6, True, []],
        ],
        inputs=[input_image, model_selector, conf_slider, fill_checkbox, class_filter],
        outputs=[output_image, metrics_output, summary_output],
        fn=predict,
        cache_examples=True,
        label="Example satellite images with aircraft",
    )

    # Connect predict button
    predict_btn.click(
        fn=predict,
        inputs=[input_image, model_selector, conf_slider, fill_checkbox, class_filter],
        outputs=[output_image, metrics_output, summary_output],
    )

    # Also allow automatic prediction on input change
    input_image.change(
        fn=predict,
        inputs=[input_image, model_selector, conf_slider, fill_checkbox, class_filter],
        outputs=[output_image, metrics_output, summary_output],
    )

    gr.Markdown(
        """
        ---
        **About:** This demo uses YOLOv26 OBB models trained on the Military Aircraft Recognition dataset.
        The models detect and classify 20 different types of military aircraft in satellite imagery.

        **Models available:** Nano (~6MB), Small (~21MB), Medium (~48MB), Large (~57MB), X (~126MB)
        """
    )


# Start the app
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
    )
