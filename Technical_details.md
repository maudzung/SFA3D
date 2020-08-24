# Super Fast and Accurate 3D Object Detection based on LiDAR Point Clouds

---

Technical details of the implementation


## 1. Input/Output & Model

- I used the ResNet-based Keypoint Feature Pyramid Network (KFPN) that was proposed in [RTM3D paper](https://arxiv.org/pdf/2001.03343.pdf). 
- The model takes a birds-eye-view RGB-map as input. The RGB-map is encoded by height, intensity and density of 3D LiDAR point clouds. 
- **Outputs**: **7 degrees of freedom** _(7-DOF)_ of objects: `(cx, cy, cz, l, w, h, θ)`
   - `cx, cy, cz`: The center coordinates.
   - `l, w, h`: length, width, height of the bounding box.
   - `θ`: The heading angle in radians of the bounding box.
- **Objects**: Cars, Pedestrians, Cyclists.

## 2. Losses function

- For main center heatmap: Used `focal loss`

- For heading angle _(direction)_: The model predicts 2 components (`imaginary value` and `real value`). 
The `im` and `re` are directly regressed by using `l1_loss`

- For `z coordinate` and `3 dimensions` (height, width, length), I used `balanced l1 loss` that was proposed by the paper
 [Libra R-CNN: Towards Balanced Learning for Object Detection](https://arxiv.org/pdf/1904.02701.pdf)

## 3. Training in details

- Set weights for the above losses are uniform (`=1.0` for all)
- Number of epochs: 300
- Learning rate scheduler: [`cosin`](https://arxiv.org/pdf/1812.01187.pdf), initial learning rate: 0.001
- Batch size: `16` (on GTX 1080Ti)

## 4. Inference

During the inference, a `3 × 3` max pooling operation is applied on the center heat map, then I keep `50` predictions whose 
center confidences are larger than 0.2.

## 5. How to expand the work

You can train the model with more classes and expand the detected area by modify configurations in the [config/kitti_dataset.py](https://github.com/maudzung/Super-Fast-Accurate-3D-Object-Detection/blob/master/src/config/kitti_config.py) 