# Quantum Fraud Detection Web Application

## 🌟 Overview

A user-friendly web interface for the Quantum Meta Fraud Detection system that allows users to input transaction details through an intuitive form and receive comprehensive fraud analysis results.

## ✨ Features

- **🎨 Modern Web Interface** - Clean, responsive design with Bootstrap 5
- **📱 Mobile Friendly** - Works perfectly on desktop, tablet, and mobile devices
- **⚡ Real-time Analysis** - Instant fraud scoring with detailed results
- **🔮 Quantum + AI Fusion** - Advanced detection using quantum computing and AI
- **📊 Visual Results** - Interactive charts, progress bars, and risk indicators
- **💾 Auto-save** - Form data automatically saved to prevent data loss
- **🎯 Quick Examples** - Pre-filled examples for testing different risk scenarios
- **📋 Comprehensive Reports** - Detailed analysis with risk factors and recommendations

## 🚀 Quick Start

### 1. Start the Web Application

```bash
# Method 1: Using the launcher (recommended)
python run_web_app.py

# Method 2: Direct start
python quantum_web_app.py

# Method 3: Using uvicorn
uvicorn quantum_web_app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access the Web Interface

Open your browser and go to: **http://localhost:8000**

The application will automatically open in your default browser when using the launcher.

### 3. Analyze Transactions

1. Fill in the transaction details in the form
2. Click "Analyze Transaction"
3. View comprehensive fraud analysis results
4. Use the results to make informed decisions

## 📋 Form Fields

### Required Fields
- **Transaction Amount** - Amount in Indian Rupees (₹)
- **Hour of Day** - Time when transaction occurred (0-23)

### Optional Fields (with smart defaults)
- **Date & Time Details**
  - Day of Week
  - Weekend Flag (auto-updated based on day)

- **User Information**
  - Sender Age Group
  - Receiver Age Group

- **Banking Details**
  - Sender State
  - Sender Bank
  - Receiver Bank

- **Technical Details**
  - Merchant Category
  - Device Type
  - Transaction Type
  - Network Type
  - Transaction Status

## 📊 Analysis Results

The web application provides comprehensive fraud analysis including:

### 🎯 Main Results
- **Fraud Score** (0-100%) with visual gauge
- **Risk Level** (MINIMAL, LOW, MEDIUM, HIGH, CRITICAL)
- **Recommended Action** with color-coded alerts
- **Confidence Score** and **Model Agreement**

### 🧠 Individual Model Scores
- Quantum Score
- Classical Score
- Neuro-QKAD Fusion Score
- Gemini Logical Score

### ⚠️ Risk Analysis
- **Primary Risk Factors** identified
- **Fraud Type** detected
- **Uncertainty Measure**
- **Processing Time**

### 📈 Visual Elements
- Circular fraud score gauge with dynamic colors
- Progress bars for individual model scores
- Color-coded risk level indicators
- Interactive charts and metrics

## 🎨 User Interface Features

### 🔧 Smart Form Features
- **Auto-completion** - Smart defaults for faster input
- **Validation** - Real-time form validation
- **Auto-save** - Form data saved automatically
- **Quick Examples** - One-click form filling for testing
- **Responsive Design** - Works on all screen sizes

### 📱 Mobile Optimization
- Touch-friendly interface
- Optimized layouts for small screens
- Fast loading and smooth interactions
- Accessible design following web standards

### 🎯 Quick Test Examples

#### Low Risk Example
- Amount: ₹2,500
- Time: 2:00 PM on Tuesday
- Expected Result: LOW risk, ~20% fraud score

#### High Risk Example  
- Amount: ₹95,000
- Time: 3:00 AM on Saturday
- Expected Result: HIGH risk, ~70% fraud score

## 🛠️ Technical Details

### Architecture
- **Backend**: FastAPI with Jinja2 templates
- **Frontend**: Bootstrap 5 + Custom CSS/JavaScript
- **AI Engine**: Quantum Meta Model integration
- **Styling**: Modern CSS with animations and transitions

### File Structure
```
├── quantum_web_app.py          # Main web application
├── run_web_app.py             # Launcher script
├── templates/
│   ├── base.html              # Base template
│   ├── index.html             # Home page with form
│   ├── results.html           # Results page
│   └── error.html             # Error page
├── static/
│   ├── style.css              # Custom styles
│   └── script.js              # JavaScript functionality
└── test_web_app.py            # Test script
```

### Dependencies
- FastAPI
- Jinja2
- Uvicorn
- Bootstrap 5 (CDN)
- Font Awesome (CDN)

## 🧪 Testing

### Run Tests
```bash
python test_web_app.py
```

### Manual Testing
1. Visit http://localhost:8000
2. Try the quick examples (Low Risk / High Risk)
3. Fill in custom transaction details
4. Verify results display correctly
5. Test on different devices/browsers

## 🔧 Configuration

### Port Configuration
Change the port in `quantum_web_app.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8080)  # Change to desired port
```

### Styling Customization
Edit `static/style.css` to customize:
- Colors and themes
- Layout and spacing
- Animations and effects
- Responsive breakpoints

### Form Customization
Modify `templates/index.html` to:
- Add/remove form fields
- Change validation rules
- Update dropdown options
- Customize layout

## 📈 Performance

- **Page Load Time**: < 2 seconds
- **Analysis Time**: 200-500ms per transaction
- **Mobile Performance**: Optimized for 3G+ networks
- **Browser Support**: Modern browsers (Chrome, Firefox, Safari, Edge)

## 🔒 Security Features

- **Input Validation** - All inputs validated on client and server
- **XSS Protection** - Template escaping prevents XSS attacks
- **CSRF Protection** - Form tokens prevent CSRF attacks
- **Rate Limiting** - Built-in protection against abuse

## 🎯 Use Cases

### Financial Institutions
- Real-time transaction monitoring
- Customer service fraud checks
- Risk assessment workflows
- Compliance reporting

### E-commerce Platforms
- Payment fraud detection
- Order verification
- Customer risk profiling
- Chargeback prevention

### Fintech Applications
- Mobile payment security
- Digital wallet protection
- P2P transfer monitoring
- Merchant verification

## 🚀 Deployment

### Local Development
```bash
python run_web_app.py
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn quantum_web_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using Docker
docker build -t quantum-fraud-web .
docker run -p 8000:8000 quantum-fraud-web
```

### Environment Variables
```bash
export GEMINI_API_KEY="your-api-key"
export MODEL_PATH="/path/to/models"
export LOG_LEVEL="INFO"
```

## 📱 Screenshots

### Home Page
- Clean, modern interface with transaction form
- Smart field organization and validation
- Quick example buttons for testing

### Results Page
- Visual fraud score gauge
- Detailed risk analysis
- Individual model breakdowns
- Actionable recommendations

### Mobile View
- Responsive design for all screen sizes
- Touch-optimized interface
- Fast loading and smooth scrolling

## 🔄 Updates and Maintenance

### Regular Updates
- Model improvements and retraining
- UI/UX enhancements
- Security patches
- Performance optimizations

### Monitoring
- Application health checks
- Performance metrics
- Error tracking
- User analytics

## 📞 Support

### Troubleshooting
1. **Server won't start**: Check if port 8000 is available
2. **Form not submitting**: Verify all required fields are filled
3. **Slow analysis**: Check model file availability
4. **Display issues**: Clear browser cache and refresh

### Getting Help
- Check the console logs for error messages
- Verify all dependencies are installed
- Test with the provided examples first
- Review the test script output

## 🎉 Success Metrics

The web application successfully provides:
- ✅ **User-friendly interface** for fraud detection
- ✅ **Real-time analysis** with comprehensive results
- ✅ **Mobile-responsive design** for all devices
- ✅ **Professional presentation** of complex AI results
- ✅ **Intuitive workflow** from input to decision

Perfect for both technical and non-technical users who need powerful fraud detection capabilities through an easy-to-use web interface!