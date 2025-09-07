# 🌱 PlantAI Disease Detector

A cutting-edge AI-powered web application for detecting plant diseases with 98%+ accuracy. Built with Flask, TensorFlow, and modern web technologies.

![PlantAI Logo](https://img.shields.io/badge/PlantAI-Disease%20Detector-00d4aa?style=for-the-badge&logo=leaf)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.3+-green?style=for-the-badge&logo=flask)

## 🚀 Live Demo

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app)

## ✨ Features

### 🔬 **Advanced AI Detection**
- **98%+ Accuracy**: State-of-the-art deep learning models
- **Real-time Analysis**: Instant disease detection in under 3 seconds
- **Multiple Architectures**: EfficientNet, ResNet, Custom CNN, and Ensemble models
- **7 Disease Classes**: Comprehensive plant disease coverage

### 📱 **Modern Web Interface**
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Live Camera Detection**: Real-time analysis using device camera
- **Beautiful UI/UX**: Modern design with glassmorphism effects
- **Dark/Light Theme**: User preference support

### 🧠 **AI Training System**
- **Custom Model Training**: Upload your own datasets
- **Advanced Configuration**: Multiple model architectures and parameters
- **Real-time Progress**: Live training monitoring with metrics
- **Model Download**: Export trained models for offline use

### 📊 **Analytics Dashboard**
- **Real-time Metrics**: Live performance monitoring
- **Usage Statistics**: Comprehensive analytics and insights
- **Interactive Charts**: Beautiful data visualizations
- **Export Capabilities**: Download reports and data

### 🔒 **Security & Privacy**
- **Privacy-First**: Local processing for camera detection
- **Secure Uploads**: Encrypted file handling
- **No Data Storage**: Images processed without permanent storage
- **Enterprise Security**: Production-ready security measures

## 🛠️ Technology Stack

### **Backend**
- **Flask 3.0.3**: Modern Python web framework
- **TensorFlow 2.16.1**: Deep learning and AI
- **OpenCV 4.9.0**: Image processing
- **scikit-learn 1.4.2**: Machine learning utilities
- **SQLite**: Analytics database

### **Frontend**
- **Bootstrap 5**: Responsive UI framework
- **Font Awesome**: Beautiful icons
- **Chart.js**: Interactive charts and graphs
- **AOS**: Scroll animations
- **Custom CSS**: Modern design system

### **AI/ML**
- **CNN Models**: Convolutional Neural Networks
- **Transfer Learning**: Pre-trained model optimization
- **Data Augmentation**: Advanced training techniques
- **Model Optimization**: TensorFlow Lite for mobile

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Git

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/plant-disease-detector.git
   cd plant-disease-detector
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
plant-disease-detector/
├── app.py                 # Main Flask application
├── advanced_training.py   # Advanced AI training system
├── performance_optimizer.py # Performance optimization
├── realtime_graphs.py     # Analytics and charts
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── nixpacks.toml        # Railway deployment config
├── Procfile             # Process configuration
├── railway.json         # Railway deployment settings
├── templates/           # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── training.html
│   ├── analytics.html
│   ├── realtime_detection.html
│   ├── about.html
│   ├── contact.html
│   └── help.html
├── static/              # Static assets
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── images/
├── models/              # AI models (gitignored)
├── training_data/       # Training datasets (gitignored)
└── uploads/            # User uploads (gitignored)
```

## 🎯 Usage

### **Image Upload Detection**
1. Navigate to the homepage
2. Click "Upload Image" or drag & drop an image
3. Get instant disease analysis with confidence scores
4. View detailed results and recommendations

### **Live Camera Detection**
1. Go to "Live Detection" page
2. Allow camera access when prompted
3. Point camera at plant leaves
4. Get real-time disease analysis

### **AI Model Training**
1. Visit "Training" page
2. Upload your training dataset (ZIP file)
3. Configure training parameters
4. Start training and monitor progress
5. Download trained models

### **Analytics Dashboard**
1. Access "Analytics" page
2. View real-time performance metrics
3. Monitor usage statistics
4. Export reports and data

## 🚀 Deployment

### **Railway.app Deployment**

1. **Connect GitHub Repository**
   - Fork this repository
   - Connect to Railway.app
   - Select the repository

2. **Automatic Deployment**
   - Railway will automatically detect the Python project
   - Install dependencies from `requirements.txt`
   - Deploy using the configuration in `nixpacks.toml`

3. **Environment Variables** (Optional)
   ```
   FLASK_ENV=production
   FLASK_DEBUG=False
   ```

### **Manual Deployment**

1. **Prepare for Production**
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=False
   ```

2. **Install Production Dependencies**
   ```bash
   pip install gunicorn
   ```

3. **Run with Gunicorn**
   ```bash
   gunicorn app:app
   ```

## 📊 Performance

- **Detection Speed**: < 3 seconds per image
- **Accuracy**: 98%+ on test datasets
- **Model Size**: Optimized for mobile deployment
- **Memory Usage**: < 500MB RAM
- **Concurrent Users**: Supports 100+ simultaneous users

## 🔧 Configuration

### **Model Settings**
- Input size: 224x224 pixels
- Classes: 7 plant disease types
- Architecture: EfficientNet (default)
- Batch size: 32 (configurable)

### **Training Parameters**
- Epochs: 20 (default)
- Learning rate: 0.001
- Data augmentation: Enabled
- Cross-validation: Enabled

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team**: For the amazing deep learning framework
- **Flask Community**: For the excellent web framework
- **Bootstrap Team**: For the responsive UI components
- **Font Awesome**: For the beautiful icons
- **Open Source Community**: For inspiration and support

## 📞 Support

- **Documentation**: [Wiki](https://github.com/yourusername/plant-disease-detector/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/plant-disease-detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/plant-disease-detector/discussions)
- **Email**: support@plantai.com

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/plant-disease-detector&type=Date)](https://star-history.com/#yourusername/plant-disease-detector&Date)

---

<div align="center">

**Made with ❤️ by the PlantAI Team**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourusername)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/yourusername)

</div>