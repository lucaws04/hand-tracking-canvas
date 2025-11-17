# ✍️ **Interactive Hand-Drawn Canvas with MediaPipe**

Welcome to the **Interactive Hand-Drawn Canvas** project! 🖼️🎨 This application uses MediaPipe for hand tracking and OpenCV for real-time drawing, offering you a unique and fun way to paint with your hands! Whether you're creating artwork or just doodling for fun, this tool is a fantastic way to engage with technology creatively.

## 🚀 **Features**

- **Hand Tracking:** Track your hand movements in real-time using MediaPipe to draw on the canvas. 🖐️
- **Resizable Brush:** Adjust the size of your brush with a pinch gesture, giving you full control over your art. ✍️
- **Color Palette:** Select from 6 colors via keyboard shortcuts (W/S to cycle through colors or click on color squares). 🌈
- **Cursor Support:** Use a cool on-screen cursor image that follows your hand movement. 🖱️
- **Clear Canvas:** Instantly clear your canvas and start fresh with a simple keyboard shortcut. 🧽
- **Responsive UI:** Scales appropriately based on window size, ensuring your art looks great at any resolution! 💻

## 🛠 **Tech Stack**

- **OpenCV**: For image processing and video capture.
- **MediaPipe**: For hand tracking and gesture recognition.
- **NumPy**: For handling array-based operations.

## 📦 **Installation**

### Prerequisites

Before running the project, you’ll need Python 3.10.16 and pip installed.

### Steps to Get Started

1. **Clone the repository**:
    ```bash
    git clone https://github.com/lucaws04/hand-tracking-canvas.git
    cd hand-drawn-canvas
    ```

2. **Install dependencies**:
    Run the following command to install all the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:
    Launch the canvas application:
    ```bash
    python canvas.py
    ```

    Your webcam will open, and you can start drawing!

---

## 🎮 **Controls**

- **Left Mouse Click**: Use the mouse to select a color or start drawing on the canvas.
- **W/S**: Cycle through the color palette (up/down to change colors).
- **D**: Toggle drawing mode (start/stop drawing).
- **C**: Clear the canvas and reset it to white.
- **Esc**: Exit the application.

<!-- ## 🖼 **Screenshot** 

![Canvas Screenshot](./assets/screenshot.png)  
_Enjoy drawing!_ -->

## 🤖 **How It Works**

- **Hand Tracking**: MediaPipe’s hand tracking API is used to detect your hand and track its movement in real-time. This allows you to draw and interact with the canvas naturally.
- **Brush Size**: The brush size dynamically adjusts based on the distance between your thumb and index finger, giving you the freedom to draw in fine detail or broad strokes.
- **Color Palette**: You can pick from 6 predefined colors, cycle through them using the keyboard, or click directly on the color palette displayed on the right side of the canvas.

---

## 🎯 **Future Improvements**

- **More Color Options**: Add the ability to create custom colors.
- **Shape Drawing**: Add support for drawing shapes (circles, squares, etc.).
- **Zoom Functionality**: Implement zoom-in/out features for detailed drawing.

---

## 🧑‍💻 **Built With**

- **Python 3.10.16**
- **OpenCV**
- **NumPy**
- **MediaPipe**

---


## 🙌 **Acknowledgements**

- Thanks to **Google MediaPipe** for providing an amazing hand-tracking library!
- OpenCV for being the go-to solution for computer vision tasks.
