# Syrian Currency Detector — قارئ العملة

An **Android** application that uses the phone camera and an on‑device **YOLOv8 TensorFlow Lite** model to detect **Syrian banknotes in real time** and announce their denomination out loud. It is built as an **accessibility / assistive tool** — when a note is recognized, the app speaks the value in Arabic, vibrates, and plays a confirmation tone, helping blind and visually‑impaired users identify cash without assistance.

The app currently recognizes three denominations: **500, 1000, and 2000 Syrian Pounds (SYP)**.

---

## How It Works

1. The camera streams frames through `Camera2` (with a legacy `Camera` fallback).
2. Each frame is cropped to **224×224**, normalized, and converted to a `ByteBuffer`.
3. The buffer is run through the bundled **YOLOv8** `.tflite` model via the TensorFlow Lite interpreter.
4. The raw model output is decoded (xywh→xyxy box conversion, confidence/class thresholding, and **Non‑Max Suppression**) into a list of recognitions.
5. The top recognition's label (`500` / `1000` / `2000`) drives the feedback in [`showResults()`](app/src/main/java/org/tensorflow/lite/examples/classification/CameraActivity.java):
   - **Audio** — plays the matching Arabic voice clip (`fivehundred.mp3`, `onethousand.mp3`, `twothousand.mp3`).
   - **Vibration** — a short haptic pulse.
   - **Tone** — a CDMA confirmation beep.
   - **Text** — the denomination is shown on screen.

---

## Features

- **Real‑time banknote detection** from the live camera feed.
- **Spoken Arabic feedback** for each detected denomination (assistive for the visually impaired).
- **Haptic + tone confirmation** on every detection.
- **On‑device inference** — the TFLite model is bundled in assets; no network/cloud is required and detection works offline.
- **Settings screen** to tune the inference runtime: model precision (Quantized / Float), compute device (CPU / GPU / NNAPI), and thread count.
- **Arabic UI** (app name "قارئ العملة" — "Currency Reader").

---

## Tech Stack

| Area             | Technology                                              |
|------------------|---------------------------------------------------------|
| Platform         | Android (min SDK 27, target/compile SDK 33)             |
| Language         | Java                                                    |
| ML runtime       | TensorFlow Lite 2.10 (+ GPU delegate, TFLite Support)   |
| Model            | YOLOv8 object detector exported to `.tflite` (float16)  |
| Camera           | Camera2 API (with legacy Camera fallback)               |
| UI               | AndroidX AppCompat, Material, ConstraintLayout          |
| Build            | Gradle (wrapper included)                               |

---

## Project Structure

```
Syrian-Currency-detector-Application/
├── app/
│   ├── build.gradle                 # App module config & dependencies
│   ├── download.gradle              # Model download helper
│   └── src/main/
│       ├── AndroidManifest.xml      # Permissions (camera, flashlight, vibrate) & activities
│       ├── assets/
│       │   ├── best_float16.tflite  # YOLOv8 detection model (used by default)
│       │   ├── best_float16n.tflite # Alternate model
│       │   └── labels.txt           # Class labels: 500 / 1000 / 2000
│       ├── java/.../classification/
│       │   ├── ClassifierActivity.java     # Launcher: frame preprocessing & detection loop
│       │   ├── CameraActivity.java         # Base activity: camera, model init, audio/haptic feedback
│       │   ├── CameraConnectionFragment.java        # Camera2 plumbing
│       │   ├── LegacyCameraConnectionFragment.java  # Legacy Camera fallback
│       │   ├── SettingsActivity.java       # Runtime settings
│       │   ├── tflite/
│       │   │   ├── Yolo.java         # YOLO post-processing base (NMS, reshape, IoU)
│       │   │   ├── Yolov8.java       # YOLOv8-specific output decoding
│       │   │   └── Recognition.java  # Detection result model
│       │   ├── env/                  # Helpers: ImageUtils, Logger, BorderedText
│       │   └── customview/           # Custom views (e.g. AutoFitTextureView)
│       └── res/
│           ├── raw/                  # fivehundred / onethousand / twothousand .mp3 voice clips
│           ├── layout/ drawable/ values/ ...
│           └── ...
├── build.gradle  settings.gradle  gradle.properties  gradlew(.bat)
└── README.md
```

### Key classes

- **`ClassifierActivity`** — the launcher activity. Preprocesses each camera frame (crop to 224×224, normalize to `[0,1]`) and calls the detector, then renders results.
- **`CameraActivity`** — base activity that owns the camera lifecycle, instantiates the `Yolov8` detector, and provides the **audio / vibration / tone feedback** in `showResults()`.
- **`Yolo` / `Yolov8`** — load the `.tflite` model and decode its output: box conversion, class/confidence thresholding, and Non‑Max Suppression into `Recognition` objects.

---

## Getting Started

### Prerequisites

- **Android Studio** (recent version)
- Android SDK 33, with a device/emulator on **Android 8.1 (API 27)+**
- A **physical device with a camera** is recommended (real‑time detection needs the camera).

### Build & Run

1. **Clone** the repository:
   ```bash
   git clone https://github.com/bishr2000/Syrian-Currency-detector-Application.git
   ```

2. **Open** the project in Android Studio and let Gradle sync.
   > `local.properties` references a local SDK path — Android Studio will regenerate it for your machine if needed.

3. **Connect** an Android device (with USB debugging) or start an emulator with a camera.

4. **Run** the `app` configuration. On first launch, **grant the camera permission**.

5. **Point the camera at a Syrian banknote** — when detected, the app announces the value (audio + vibration + on‑screen text).

The YOLOv8 model (`best_float16.tflite`) and the audio clips ship inside the app, so no extra setup or network connection is required.

---

## Notes & Limitations

- Recognition is limited to the three labels in `labels.txt` (**500 / 1000 / 2000 SYP**); other notes/coins are not detected.
- Detection quality depends on lighting, focus, and how the note fills the frame — good lighting and a steady, framed shot work best.
- Two model files are bundled; `best_float16.tflite` is the one wired up in `CameraActivity` (`MODEL_PATH`).
- The project is based on the TensorFlow Lite Android image‑classification example, adapted for YOLOv8 detection and Syrian currency.

---

## License

No license file is currently included. Add one (e.g. Apache‑2.0, matching the upstream TensorFlow examples) if you intend others to reuse this code.

## Author

**bishr2000**
