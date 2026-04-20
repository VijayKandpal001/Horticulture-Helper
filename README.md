# 🌿 Horticulture Helping Hand

A comprehensive AI-powered plant disease prediction and treatment recommendation system that leverages deep learning and LLM technology to assist farmers and gardeners in identifying and managing crop diseases.

## 📋 Project Overview

Horticulture Helping Hand is an intelligent system designed to:
- **Detect plant diseases** from leaf images using a trained deep learning model
- **Classify diseases** with confidence scores
- **Provide treatment recommendations** using Large Language Models (LLM)
- **Offer both organic and chemical solutions** tailored to specific diseases
- **Support prevention strategies** for sustainable farming

## ✨ Features

- 🖼️ **Image-Based Disease Detection** - Upload leaf images and get instant predictions
- 🤖 **AI-Powered Recommendations** - LLM-generated treatment plans with multiple options
- 💚 **Organic & Chemical Solutions** - Both sustainable and conventional treatment approaches
- 📊 **Confidence Scoring** - Know how confident the model is about predictions
- 🎨 **User-Friendly Interface** - Streamlit web app for easy accessibility
- 🔌 **RESTful API** - FastAPI backend for programmatic access
- 🐳 **Docker Support** - Easy deployment with containerization
- ☁️ **Cloud Ready** - Pre-configured for deployment on Railway/Cloud platforms

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **TensorFlow/Keras** - Deep learning framework for model inference
- **Python 3.12** - Core programming language
- **Pillow** - Image processing

### Frontend
- **Streamlit** - Interactive web framework for data apps
- **Requests** - HTTP client for API communication

### Model & ML
- **Pre-trained CNN Model** - Fruit/Plant disease classification
- **LLM Integration** - For generating detailed treatment recommendations
- **Class Mapping** - JSON-based disease classification

### Deployment
- **Docker** - Containerization (Python 3.12-slim base)
- **Railway** - Cloud deployment platform (or similar)
- **Uvicorn** - ASGI server for FastAPI

## 📁 Project Structure

```
HorticultureHelpingHand/
├── app/
│   ├── api.py                           # FastAPI application & endpoints
│   ├── main.py                          # Streamlit frontend application
│   ├── llm_service.py                   # LLM service for recommendations
│   ├── requirements.txt                 # Python dependencies
│   ├── class_indices.json               # Disease class mapping
│   └── trained_model/
│       └── fruits_disease_prediction_model_updated.h5  # Pre-trained model
├── Dockerfile.api                       # Docker config for API service
├── Dockerfile.streamlit                 # Docker config for Streamlit app
├── model.ipynb                          # Jupyter notebook for model development
├── response_sample.json                 # Sample API response example
├── README.md                            # Project documentation
└── papers/                              # Research papers & documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- Docker (optional, for containerization)

### Local Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd HorticultureHelpingHand
```

2. **Create a virtual environment** (recommended)
```bash
# Using venv
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. **Install dependencies**
```bash
cd app
pip install -r requirements.txt
```

## 💻 Usage

### Option 1: Streamlit Frontend (Recommended for Users)

Run the interactive Streamlit web application:
```bash
cd app
streamlit run main.py
```

The application will open at `http://localhost:8501`

**Steps to use:**
1. Upload an image of a diseased leaf
2. The system will predict the disease and confidence score
3. Get detailed treatment recommendations (organic & chemical)
4. View prevention strategies

### Option 2: FastAPI Backend (For Developers/Integration)

Start the API server:
```bash
cd app
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

Access the interactive API documentation at `http://localhost:8000/docs` (Swagger UI)

## 🔌 API Endpoints

### 1. **Health Check**
```
GET /
```
**Response:**
```json
{
  "message": "Plant Disease Prediction API is running"
}
```

### 2. **Disease Prediction**
```
POST /predict
```
**Parameters:**
- `file` (FormData): Image file of the affected plant leaf

**Response:**
```json
{
  "prediction": "Tomato Late Blight",
  "confidence": 0.95
}
```

### 3. **Treatment Recommendation**
```
GET /treatment?disease=Tomato%20Late%20Blight&confidence_score=0.95
```

**Parameters:**
- `disease` (string): Disease name from prediction
- `confidence_score` (float): Confidence score of the prediction

**Response:**
```json
{
  "disease_confirmation": "Confirmed",
  "description": "Detailed disease description...",
  "organic_treatment": ["Treatment option 1", "Treatment option 2"],
  "chemical_treatment": ["Chemical option 1", "Chemical option 2"],
  "prevention": ["Prevention strategy 1", "Prevention strategy 2"],
  "note": "Additional important notes..."
}
```

## 🤖 Model Details

### Architecture
- **Model Type**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224 RGB images
- **Pre-trained**: Custom trained on plant disease dataset
- **File**: `app/trained_model/fruits_disease_prediction_model_updated.h5`

### Image Preprocessing
- Resize to 224x224 pixels
- Convert to RGB format
- Normalize pixel values (0-1 range)
- Expand dimensions for batch processing

### Supported Diseases
See `app/class_indices.json` for the complete list of detectable plant diseases and fruits.

## 🐳 Docker Deployment

### Build Docker Images

**API Service:**
```bash
docker build -f Dockerfile.api -t horticulture-api:latest .
```

**Streamlit Service:**
```bash
docker build -f Dockerfile.streamlit -t horticulture-streamlit:latest .
```

### Run Containers

**API:**
```bash
docker run -p 8000:8000 -e PORT=8000 horticulture-api:latest
```

**Streamlit:**
```bash
docker run -p 8501:8501 -e PORT=8501 horticulture-streamlit:latest
```

## ☁️ Deployment on Cloud

### Railway.app (Recommended)
1. Push to GitHub
2. Connect repository to Railway
3. Set environment variables (PORT, API_URL)
4. Deploy from Dockerfile

### Environment Variables
- `API_URL`: URL of the backend API (for Streamlit app)
- `PORT`: Port number (default: 8000 for API, 8501 for Streamlit)

## 📊 Example Workflow

1. **User uploads an image** → Streamlit frontend
2. **Image preprocessing** → Resizing & normalization
3. **Model prediction** → Disease classification + confidence
4. **API call** → `/treatment` endpoint with disease info
5. **LLM processing** → Generates detailed recommendations
6. **Display results** → Shows disease info, treatments, & prevention

## 🔧 Development

### Project Setup for Development
```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests (if available)
pytest

# Code formatting
black app/
```

### Adding New Features
- Model updates: Retrain and save to `trained_model/`
- New endpoints: Add to `app/api.py`
- UI improvements: Modify `app/main.py`
- Treatment logic: Update `app/llm_service.py`

## 📚 Model Training

Refer to `model.ipynb` for:
- Data loading and exploration
- Model architecture design
- Training pipeline
- Model evaluation metrics
- Class mapping generation

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature/YourFeature`)
6. Open a Pull Request

## ⚠️ Important Notes

- **Model Accuracy**: While the model provides reliable predictions, always verify with domain experts
- **Disease Confirmation**: Treatment recommendations are generated based on predicted diseases
- **Environmental Factors**: Recommendations consider common farming practices and conditions
- **Regular Updates**: Keep the model updated with new disease data for better accuracy

## 📝 License

This project is provided as-is. Please check the LICENSE file for detailed information.

## 🙏 Acknowledgments

- Plant disease dataset sources
- TensorFlow and Keras communities
- Streamlit for the excellent UI framework
- FastAPI for the robust API framework
- LLM providers for recommendation generation

## 📞 Support & Contact

For issues, questions, or suggestions, please:
- Open an issue on GitHub
- Check existing documentation in `/papers` folder
- Review API documentation at `/docs` endpoint

---

**Made with ❤️ for farmers and gardeners worldwide**

Last Updated: April 2026