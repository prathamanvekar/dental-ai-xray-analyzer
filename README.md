# DentalAI: Integrated X-Ray Analysis System

## Project Overview

DentalAI is an AI-powered dental X-ray analysis system that uses advanced computer vision and deep learning to automate dental diagnosis. The system leverages YOLO-based object detection and CNN-BiLSTM neural networks to identify teeth, classify jaw types, detect tooth types, and diagnose dental diseases from dental radiographs.

### Key Features

- **Automated Tooth Detection**: YOLO-based object detection to locate and identify individual teeth in X-ray images
- **Jaw Classification**: Distinguish between Mandible (lower jaw) and Maxilla (upper jaw)
- **Tooth Type Classification**: Classify teeth into Canine, Incisor, Molar, or Premolar categories
- **Disease Detection**: Identify 7 dental conditions:
  - Caries (tooth decay)
  - Cyst
  - Impacted Tooth
  - Periapical Lesion
  - Periodontal Bone Loss
  - Root Canal Treated
  - Healthy
- **Interactive GUI**: User-friendly Tkinter interface for loading and analyzing X-ray images
- **Detailed Reports**: Generates clinical analysis reports with findings

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster inference)
- 2GB+ free disk space for models

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dental-ai-xray-analyzer.git
cd dental-ai-xray-analyzer
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download/Prepare Models

Place the pre-trained model files in the `models/` directory:

```
models/
├── best_yolo.pt              # YOLO object detection model
├── disease_model.pth         # Disease classification model
├── jaw_model.pth             # Jaw type classification model
└── tooth_model.pth           # Tooth type classification model
```

## Usage

### Running the GUI Application

Launch the interactive desktop application:

```bash
python main.py
```

**Steps to analyze an X-ray:**

1. Click "📂 Load X-Ray" to select a dental X-ray image
2. Click "⚡ Analyze Image" to run the analysis
3. View detection results, classifications, and disease predictions
4. Save the analysis report if needed

### Running Inference Programmatically

```python
from inference import DentalInferencePipeline

# Initialize the pipeline
pipeline = DentalInferencePipeline(models_dir='models')

# Run inference on an image
results = pipeline.infer_image('path/to/xray.jpg')

# Access results
print(results)
```

## Project Structure

```
dental-ai-xray-analyzer/
├── main.py                    # Tkinter GUI application
├── inference.py               # Core inference pipeline
├── models_arch.py             # CNN-BiLSTM model architecture
├── rule_engine.py             # Clinical decision rules
├── requirements.txt           # Python dependencies
├── data/                      # Sample/training data (if available)
├── models/                    # Pre-trained model files
│   ├── best_yolo.pt
│   ├── disease_model.pth
│   ├── jaw_model.pth
│   └── tooth_model.pth
├── README.md                  # This file
└── .gitignore                 # Git ignore patterns
```

## Model Architecture

### CNN-BiLSTM Classifier

The classification models use a hybrid architecture:

- **Backbone**: EfficientNet (feature extraction)
- **Recurrent Layer**: Bidirectional LSTM (temporal/spatial pattern recognition)
- **Dropout**: 0.5 (regularization)

This architecture enables robust classification across multiple dental conditions with 224x224 input images.

## Supported File Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## System Requirements

### Hardware

- **CPU**: Dual-core processor minimum
- **GPU**: NVIDIA GPU with CUDA support (recommended for real-time analysis)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 3GB+ for models and application

### Software

- Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- Python 3.8 or higher

## Troubleshooting

### Error: "Failed to load models"

- Ensure all model files are in the `models/` directory
- Verify file names match exactly: `best_yolo.pt`, `disease_model.pth`, `jaw_model.pth`, `tooth_model.pth`
- Check that model files are not corrupted

### Error: "CUDA out of memory"

- Reduce image resolution or batch size
- Switch to CPU inference if GPU memory is limited
- Close other GPU-consuming applications

### GUI Not Responding

- Ensure images are properly formatted (JPEG/PNG)
- Try with smaller image sizes (< 5MB)
- Check Python and Tkinter installation

## Performance Notes

- **Inference Time**: 1-3 seconds per image (GPU-accelerated)
- **Accuracy**: Varies by model training; evaluate on your dataset
- **Batch Processing**: Not currently supported; process images one at a time

## License

[Add your license here - e.g., MIT, Apache 2.0]

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## Citation

If you use this project in your research, please cite:

```
[Add citation information]
```

## Contact

For issues, questions, or suggestions, please open an issue on GitHub or contact the maintainers.

## Disclaimer

This tool is for educational and research purposes. It should not be used for clinical diagnosis without validation by qualified dental professionals. Always consult with a licensed dentist for medical advice.
