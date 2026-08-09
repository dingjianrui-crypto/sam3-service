# Paddle Blade Completeness Classification

## 1. Objective

The goal is to train a neural network that receives a **segmented paddle mask** and predicts whether the paddle blade is:

- **Complete / well formed**
- **Cropped / incomplete**

The model does **not** need to determine why the blade is cropped. In particular:

- Water itself is not the classification target.
- Human-body occlusion of the middle shaft is not important.
- The model does not need to reconstruct the full paddle.
- The model does not need to predict a full paddle mask.
- The main signal is whether the visible blade geometry is consistent with a valid, complete paddle blade.

Recommended binary labels:

```text
0 = complete / well paddle
1 = cropped / incomplete blade
```

The network should output a probability:

```text
P(cropped)
```

rather than only a hard binary result.

---

# 2. Source Data

Use real video frames as the source dataset.

For every frame:

```text
Video frame
    ↓
Paddle segmentation model
    ↓
Paddle mask + bounding box
    ↓
Training sample preparation
```

The segmentation model should provide:

- Binary paddle mask
- Paddle bounding box

The classifier should primarily operate on the **mask**, rather than the original full RGB image.

---

# 3. Recommended Input

## 3.1 Use a Mask-Only Crop

Recommended model input:

```text
1 × H × W binary paddle mask
```

For example:

```text
1 × 256 × 256
```

where:

```text
0 = background
1 = detected paddle
```

Advantages:

- Removes water appearance.
- Removes boat appearance.
- Removes background variation.
- Removes most athlete appearance.
- Removes color and lighting dependence.
- Forces the classifier to focus on paddle geometry.
- Makes the model easier to train with less data.

---

## 3.2 Crop Using the Paddle Bounding Box

Do not use the whole video frame.

Pipeline:

```text
Full frame
    ↓
Paddle mask + bounding box
    ↓
Crop around paddle bounding box
    ↓
Add small padding
    ↓
Resize/pad to fixed input size
```

Recommended padding:

```text
~10–20% around the paddle bounding box
```

The purpose of padding is to avoid clipping paddle pixels because of small segmentation or bounding-box errors.

---

## 3.3 Preserve Aspect Ratio

Do not stretch the paddle into a square.

For example:

```text
Original crop
    ↓
Resize while preserving aspect ratio
    ↓
Zero-pad to 256 × 256
```

Example:

```text
+----------------------------+
|                            |
|          paddle            |
|             /              |
|            /               |
|           /                |
|         █████              |
|        ███████             |
|                            |
+----------------------------+
```

This keeps the paddle's real geometric proportions.

---

# 4. Do Not Normalize Blade Roll

A paddle blade changes its 2D silhouette when the athlete rotates the paddle around its shaft.

A complete blade may therefore look very different depending on shaft roll:

```text
Front-facing        Partly rotated       Nearly edge-on

  ███████               ████                 ██
 █████████              █████                 ██
███████████              ████                 ██
 █████████                ███                 ██
    ██                     ██                 ██
```

All of these may represent a **complete blade**.

Therefore, do not attempt to deform or normalize the blade into one standard symmetric shape.

The classifier must learn the **distribution of valid blade appearances**.

---

# 5. In-Plane Orientation

There are two different types of rotation.

## 5.1 In-Plane Image Rotation

Example:

```text
       /                     --------
      /
     /
```

This is simply the paddle changing orientation within the image.

For the first model version, it is recommended to **leave this variation in the training data** and use rotational augmentation.

This avoids introducing unnecessary preprocessing complexity.

---

## 5.2 Blade Roll Around the Shaft

Example:

```text
Wide blade silhouette  →  narrow blade silhouette
```

This is a real 3D appearance change and must **not** be normalized away.

The training set should contain many examples across the full blade rotation cycle.

---

# 6. Labeling the Dataset

Use two labels.

## Complete

```text
label = 0
```

Examples include:

- Blade fully visible.
- Blade rotated toward camera.
- Blade rotated partly away from camera.
- Blade almost edge-on.
- Shaft partially hidden by the athlete.
- Paddle mask split into disconnected pieces because the athlete blocks the shaft.
- Moderate motion blur.
- Minor segmentation imperfections.

The important condition is:

> The blade itself is sufficiently complete to represent a valid paddle blade.

---

## Cropped

```text
label = 1
```

Examples include:

- Part of blade disappears.
- Blade contour terminates prematurely.
- Significant blade area is missing.
- Only part of the blade remains visible.
- Paddle blade is cropped by immersion.
- Blade is otherwise physically incomplete in the segmentation result.

The classifier does not need to know what caused the missing part.

---

# 7. Human Occlusion

The middle of the paddle may frequently be hidden by the athlete.

Example:

```text
blade
 █████
  ███
   │
   │

   <- shaft hidden by athlete ->

   │
   │
```

This should still be labeled:

```text
complete
```

if the blade itself is complete.

Do **not** remove or reconstruct the human-occluded part of the shaft.

The segmentation mask already removes most irrelevant visual information.

The classifier needs to learn:

```text
Missing middle shaft ≠ cropped blade
```

---

# 8. Important Dataset Balance

Human occlusion must not accidentally correlate with the target class.

The dataset should contain all combinations:

| Shaft occluded by human | Blade complete | Label |
|---|---|---|
| No | Yes | Complete |
| Yes | Yes | Complete |
| No | No | Cropped |
| Yes | No | Cropped |

This prevents the model from learning:

```text
disconnected paddle mask = cropped
```

which would be incorrect.

---

# 9. Data Diversity

Dataset diversity is likely more important than using a complicated network.

## Complete-class examples should include:

- Paddle pointing upward.
- Paddle pointing downward.
- Paddle at different image-plane angles.
- Clockwise and anticlockwise orientations.
- Blade facing camera.
- Blade at intermediate roll angles.
- Blade nearly edge-on.
- Shaft partly occluded by human.
- Different athletes.
- Different paddle models.
- Different blade sizes.
- Different camera distances.
- Different image resolutions.
- Motion blur.
- Mild segmentation noise.
- Different videos and locations.

## Cropped-class examples should contain the same diversity:

- Different paddle angles.
- Different shaft-roll angles.
- Different athletes.
- Different camera scales.
- Different amounts of blade cropping.
- Human occlusion plus blade cropping.
- Motion blur.
- Segmentation imperfections.

A major dataset rule is:

> Every normal pose that exists in the complete class should ideally also occur in the cropped class.

For example, if most nearly edge-on blades are labeled cropped, the classifier may incorrectly learn:

```text
thin blade = cropped
```

instead of learning blade completeness.

---

# 10. Train/Validation/Test Split

Because the source data comes from videos, **do not randomly split individual frames**.

Adjacent frames are extremely similar.

A random frame split can cause almost identical images to appear in both training and validation sets, producing unrealistically high accuracy.

Instead, split by:

- Video
- Recording session
- Athlete
- Camera sequence

Example:

```text
Training:
videos 01–20

Validation:
videos 21–24

Test:
videos 25–30
```

Ideally, the test set should contain videos that were never used during training.

---

# 11. Remove Near-Duplicate Frames

Video contains many almost identical consecutive frames.

For example:

```text
frame 100
frame 101
frame 102
frame 103
```

may contain essentially the same paddle mask.

Avoid letting thousands of nearly identical frames dominate training.

Possible strategies:

```text
sample every N frames
```

or use paddle motion / mask similarity to select sufficiently different frames.

For example:

```text
keep frame if IoU(previous_mask, current_mask) < threshold
```

This increases effective dataset diversity.

---

# 12. Recommended Data Augmentation

Because the input is a binary mask, augmentation should primarily reproduce realistic geometric and segmentation variations.

## Recommended

### Random in-plane rotation

```text
0–360°
```

This makes the classifier robust to arbitrary paddle direction.

### Small translations

Move the paddle slightly inside the padded input.

### Small scale changes

Simulate different camera distances and bounding-box sizes.

### Minor morphological changes

Occasionally apply:

```text
small erosion
small dilation
```

to simulate segmentation-mask variation.

Keep these changes mild so the blade's semantic shape remains valid.

### Small mask defects

Randomly remove a few isolated pixels or small regions to simulate real segmentation errors.

### Synthetic middle-shaft occlusion

This augmentation is especially important.

Take a complete paddle mask:

```text
      █████
       ███
        │
        │
        │
        │
```

and randomly erase part of the middle shaft:

```text
      █████
       ███
        │


        │
        │
```

The label remains:

```text
complete
```

This teaches the network:

```text
shaft gap → ignore
blade incompleteness → important
```

---

# 13. Augmentations to Avoid

Avoid transformations that change the actual blade geometry unrealistically.

Do not aggressively use:

- Horizontal stretching
- Vertical stretching
- Perspective warping
- Arbitrary blade deformation
- Strong elastic deformation
- Artificial blade widening
- Artificial blade narrowing

These can accidentally turn a complete blade into an unrealistic or apparently cropped one.

---

# 14. Recommended Neural Network

Start with a small 2D CNN.

There is no need for a large ViT or segmentation model.

Recommended baseline:

```text
Input
1 × 256 × 256
        ↓
Conv 3×3, 32
BatchNorm
ReLU
MaxPool
        ↓
Conv 3×3, 64
BatchNorm
ReLU
MaxPool
        ↓
Conv 3×3, 128
BatchNorm
ReLU
MaxPool
        ↓
Conv 3×3, 256
BatchNorm
ReLU
        ↓
Global Average Pooling
        ↓
Dense 128
ReLU
Dropout
        ↓
Dense 1
        ↓
P(cropped)
```

Approximate feature sizes:

```text
1 × 256 × 256
       ↓
32 × 128 × 128
       ↓
64 × 64 × 64
       ↓
128 × 32 × 32
       ↓
256 × 32 × 32
       ↓
256
       ↓
128
       ↓
1
```

---

# 15. Why Use Global Average Pooling?

Use:

```text
AdaptiveAvgPool2d(1)
```

instead of flattening the entire spatial feature map.

Advantages:

- Reduces parameter count.
- Reduces overfitting.
- Makes the classifier less dependent on the exact blade location.
- Works well with rotational and translational variation.
- Allows the CNN to learn whether relevant shape features exist anywhere in the crop.

---

# 16. Example PyTorch Model

```python
import torch
import torch.nn as nn


class PaddleCompletenessCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

The model should return a logit.

Inference:

```python
logit = model(mask)
prob_cropped = torch.sigmoid(logit)

is_cropped = prob_cropped > 0.5
```

---

# 17. Loss Function

Use:

```python
torch.nn.BCEWithLogitsLoss()
```

Example:

```python
criterion = nn.BCEWithLogitsLoss()
```

Labels:

```text
0.0 = complete
1.0 = cropped
```

Do not place a sigmoid layer inside the network when using `BCEWithLogitsLoss`, because the loss function already handles the sigmoid operation numerically.

---

# 18. Class Imbalance

Real video may contain many more complete frames than cropped frames, or vice versa.

Monitor:

```text
number of complete samples
number of cropped samples
```

If significantly imbalanced, use one or more of:

- Balanced sampling
- Weighted loss
- Hard-negative mining
- Selective frame sampling

Do not rely only on overall accuracy.

---

# 19. Evaluation Metrics

Recommended metrics:

```text
Precision
Recall
F1 score
ROC-AUC
Confusion matrix
```

For this application, recall for the cropped class can be particularly important:

```text
cropped recall =
correctly detected cropped paddles
---------------------------------
all actually cropped paddles
```

You should also evaluate results separately for:

- Large visible blade
- Nearly edge-on blade
- Human-occluded shaft
- Strong motion blur
- Small paddle
- Different paddle models
- Different videos

This helps reveal dataset bias.

---

# 20. Probability Output

Do not immediately reduce the network output to only:

```text
complete / cropped
```

Keep:

```text
P(cropped)
```

Example:

```text
0.03 → clearly complete
0.15 → likely complete
0.48 → uncertain
0.81 → likely cropped
0.97 → clearly cropped
```

This is particularly useful when processing video.

---

# 21. Temporal Smoothing During Video Inference

Frame-by-frame classification may fluctuate:

```text
0.08
0.11
0.07
0.41
0.82
0.91
0.86
```

Apply temporal smoothing after classification.

For example:

```text
smoothed_t =
    α × prediction_t
    + (1 - α) × smoothed_(t-1)
```

or use a short median / moving-average window.

This produces a more stable paddle-state signal.

Example:

```text
Complete                     Cropped
    ↓                           ↓

0.05 → 0.08 → 0.12 → 0.35 → 0.76 → 0.91 → 0.95
```

The transition can later be useful for paddle-event analysis.

---

# 22. Optional Upgrade: ResNet-18

If the custom CNN reaches its limit and enough training data is available, the next model to test is **ResNet-18**.

Modify the first convolution from:

```text
3 input channels
```

to:

```text
1 input channel
```

and replace the final classifier with:

```text
Linear(..., 1)
```

However, the small custom CNN should be the baseline.

Recommended progression:

```text
Small CNN
    ↓
Analyze failure cases
    ↓
Improve dataset / augmentation
    ↓
ResNet-18 only if necessary
```

Dataset quality is likely to matter more than simply increasing model size.

---

# 23. Recommended End-to-End Training Pipeline

```text
Real kayak videos
       ↓
Extract representative frames
       ↓
Paddle segmentation model
       ↓
Paddle mask + bbox
       ↓
Manual / assisted labeling
       │
       ├── Complete
       └── Cropped
       ↓
Remove excessive near-duplicate frames
       ↓
Split by video/session
       │
       ├── Train
       ├── Validation
       └── Test
       ↓
Crop paddle bbox
       ↓
Add 10–20% padding
       ↓
Preserve aspect ratio
       ↓
Pad to 256 × 256
       ↓
Binary mask
       ↓
Training augmentation
       │
       ├── Random 0–360° rotation
       ├── Translation
       ├── Small scale changes
       ├── Mild erosion/dilation
       ├── Minor segmentation noise
       └── Synthetic middle-shaft occlusion
       ↓
Small CNN
       ↓
BCEWithLogitsLoss
       ↓
P(cropped)
```

---

# 24. Recommended First Experiment

Keep the initial experiment deliberately simple.

## Input

```text
1 × 256 × 256 paddle mask
```

## Classes

```text
0 = complete
1 = cropped
```

## Model

```text
4-layer CNN + Global Average Pooling
```

## Augmentation

```text
random rotation
small translation
small scale variation
mild mask noise
middle-shaft occlusion
```

## Split

```text
split by source video, never random adjacent frames
```

## Output

```text
P(cropped)
```

## Inference

```text
SAM3 paddle mask
       ↓
bbox crop
       ↓
resize + pad
       ↓
CNN
       ↓
P(cropped)
       ↓
temporal smoothing
       ↓
stable complete/cropped state
```

---

# 25. Key Design Principles

The most important principles are:

1. **Use the segmented paddle mask rather than the whole RGB frame.**
2. **Preserve real blade shape variation caused by shaft roll.**
3. **Do not force all blades into one canonical symmetric appearance.**
4. **Do not attempt to reconstruct human-occluded shaft sections.**
5. **Teach the model explicitly that middle-shaft gaps are acceptable.**
6. **Make complete and cropped classes diverse across the same paddle poses.**
7. **Split the dataset by video/session to prevent frame leakage.**
8. **Use a small CNN first; improve data before increasing model complexity.**
9. **Predict a continuous cropped probability.**
10. **Use temporal smoothing during video inference.**

The resulting classifier should learn:

```text
"Does the observed paddle mask contain a physically plausible,
complete blade for its current pose?"
```

rather than:

```text
"Does this mask match one fixed blade template?"
```
