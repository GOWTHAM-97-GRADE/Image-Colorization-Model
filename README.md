# Image Colorization with CDE-GAN \U0001F3A8

Welcome to the **Image Colorization Project**! This repository contains a PyTorch implementation of a **Cooperative Dual Evolution-Based Generative Adversarial Network (CDE-GAN)** designed to transform black-and-white images into vibrant, colorized versions. Using deep learning and GANs, this model learns to add realistic colors to grayscale images with impressive results. \U0001F308

## 🚀 Project Highlights

- **Model**: CDE-GAN (ResNet-based Generator & N-Layer Discriminator)
- **Achievements**:
  - Generator Loss: **0.3232**
  - Discriminator Loss: **0.1989**
- **Dataset**: [Image Colorization Dataset](https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset) from Kaggle
- **Outcome**: Visually stunning colorized images from grayscale inputs
- **Potential**: With more computational power and a larger dataset, even better results are possible!

## 📌 Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## 🛠 Installation

To run this project locally, follow these steps:

### 1️⃣ Clone the Repository:
```bash
git clone https://github.com/your-username/image-colorization-cde-gan.git
cd image-colorization-cde-gan
```

### 2️⃣ Install Dependencies:
Ensure you have **Python 3.8+** installed, then install the required packages:
```bash
pip install numpy torch torchvision pillow tqdm matplotlib kagglehub
```

### 3️⃣ Set Up CUDA (Optional):
If you have a GPU, ensure CUDA is installed and compatible with PyTorch for faster training. The code automatically detects and uses CUDA if available.

---

## 📂 Dataset

This project uses the **Image Colorization Dataset** from Kaggle, which is automatically downloaded via `kagglehub`. The dataset includes:

- **Train Set**: Black-and-white (`train_black`) and color (`train_color`) image pairs.
- **Test Set**: Black-and-white (`test_black`) and color (`test_color`) image pairs.

The dataset is automatically downloaded to:
```
/root/.cache/kagglehub/datasets/aayush9753/image-colorization-dataset/versions/1
```
when you run the script.

---

## 🎯 Usage

### 1️⃣ Train the Model
Run the script to train the GAN:
```bash
python colorization.py
```
- Training runs for **5 epochs** by default.
- Model checkpoints are saved as `generator7_epochX.pth` and `discriminator7_epochX.pth`.

### 2️⃣ Generate Colorized Images
After training, use the function `generate_colorized_image()` to colorize a black-and-white image:
```python
generate_colorized_image(netG, "path/to/bw_image.jpg", "colorized_output.jpg")
```
- The output is saved in the dataset directory.

---

## 🏗 Model Architecture

### 🎨 Generator (ResNet-Based)
- **Input**: Grayscale image (**1 channel**)
- **Output**: RGB image (**3 channels**)
- **Structure**:
  - Downsampling with **2 convolutional layers**
  - **9 ResNet blocks**
  - Upsampling with **transposed convolutions**
  - Final **Tanh activation**

### 🕵️‍♂️ Discriminator (N-Layer Discriminator)
- **Input**: RGB image (**3 channels**)
- **Output**: Real/Fake classification
- **Structure**:
  - Multi-layer **convolutional network with LeakyReLU**
  - **Batch normalization** for stability

### 📉 Loss Function
- **GAN Loss**: Least Squares GAN (**LSGAN**) loss for stable training.

---

## 🎓 Training

- **Epochs**: 5
- **Batch Size**: 16
- **Optimizer**: Adam (`lr=0.0002, betas=(0.5, 0.999)`) 
- **Device**: CUDA (if available) or CPU
- **Progress Monitoring**: `tqdm` and periodic loss printing

Example training output:
```
Epoch 1/5: 100%|██████████| ... [G Loss: 0.XXXX, D Loss: 0.XXXX]
...
Epoch 5 Completed. G Loss: 0.3232, D Loss: 0.1989
```

---

## 📸 Results

The model successfully adds **vibrant, realistic colors** to grayscale images. Below is an example:

https://www.linkedin.com/posts/gowtham-p-0bb3a4277_machinelearning-deeplearning-artificialintelligence-activity-7274038013842583552-GN1o?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAAEOC2Y8BwzEUsNPJJzqRP2XPkL2KxLKOYcQ

check out this!!!

---

## 🚀 Future Work

- **More Epochs**: Increase training duration for better convergence.
- **Larger Dataset**: Incorporate a more diverse dataset for improved generalization.
- **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, and ResNet blocks.
- **Advanced Architectures**: Explore U-Net or attention-based GANs for finer details.

---

## 🤝 Contributing

Feel free to **fork this repository**, submit **issues**, or create **pull requests**! Any contributions to enhance the model or documentation are welcome.

### Contribution Steps:
1. **Fork the repo**
2. **Create a new branch**:
   ```bash
   git checkout -b feature-branch
   ```
3. **Commit your changes**:
   ```bash
   git commit -m "Add feature"
   ```
4. **Push to the branch**:
   ```bash
   git push origin feature-branch
   ```
5. **Open a Pull Request**

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Happy Colorizing! 🌟
