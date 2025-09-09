# Neuro-QKAD: Quantum-Classical Fusion Fraud Detection

## Overview

Neuro-QKAD is an advanced fraud detection system that combines quantum computing, classical machine learning, and AI-powered logical analysis to provide comprehensive transaction fraud scoring. The system uses a meta-fusion approach that integrates:

- **Quantum Kernel SVM**: 4-qubit quantum feature mapping using PennyLane for quantum advantage in pattern recognition
- **Classical XGBoost**: Gradient boosting with comprehensive transaction features for baseline fraud detection
- **Rule-based Logic**: Domain-specific fraud detection patterns and heuristics
- **Gemini AI Integration**: Google's Gemini API for advanced logical fraud pattern analysis
- **Meta Fusion Model**: Logistic regression combining all approaches for enhanced accuracy

The system processes 14 different transaction features including amount, timing, user demographics, device information, and banking details to generate fraud scores from 0-100% with risk level classifications.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Components

**Frontend Architecture**
- FastAPI web framework with Jinja2 templating
- Bootstrap 5 responsive UI with custom CSS
- Interactive fraud analysis forms with real-time validation
- Results visualization with fraud score gauges and risk level indicators

**Backend Architecture**
- Multiple FastAPI applications for different deployment scenarios (simple, enhanced, meta)
- Modular design with separate components for quantum processing, classical ML, and AI integration
- Asynchronous processing capabilities for handling concurrent fraud analysis requests

**Data Processing Pipeline**
- Comprehensive preprocessing with StandardScaler for numerical features
- LabelEncoder for categorical features (age groups, device types, merchant categories)
- Feature engineering for quantum angle conversion (0 to 2π range)
- Support for both real UPI transaction data and synthetic data generation

**Quantum Computing Integration**
- PennyLane quantum framework with default.qubit simulator
- 4-qubit quantum circuits using RY rotations and CNOT entangling gates
- Quantum kernel computation using fidelity-based similarity measures
- Fallback mechanisms when quantum dependencies are unavailable

**Classical Machine Learning**
- XGBoost gradient boosting with class imbalance handling
- Support Vector Machines with quantum and classical kernels
- Logistic regression for meta-model fusion
- Comprehensive model evaluation with ROC-AUC, precision-recall metrics

**Model Persistence**
- Joblib-based model serialization and loading
- Centralized model management in `/models` directory
- Graceful fallback when pre-trained models are unavailable

### Design Patterns

**Meta-Learning Fusion**
The system implements a sophisticated meta-learning approach where individual models (quantum, classical, rule-based, AI) generate independent fraud scores that are then combined using a trained fusion model. This provides better accuracy than any single approach.

**Modular Architecture**
Clear separation of concerns with dedicated modules for data preprocessing (`data_prep.py`), quantum kernels (`qkernel.py`), model training (`train.py`), and persistence (`save_load.py`).

**Multiple Deployment Configurations**
The codebase supports various deployment scenarios from simple demos to production-ready applications with different feature sets and complexity levels.

**Error Handling and Resilience**
Comprehensive error handling with fallback mechanisms for missing dependencies, failed API calls, and model loading errors.

## External Dependencies

### Core Web Framework
- **FastAPI 0.104.1**: Modern web framework for building APIs with automatic OpenAPI documentation
- **Uvicorn**: ASGI server for running FastAPI applications
- **Jinja2**: Templating engine for HTML generation
- **Python-multipart**: For handling form data and file uploads

### Machine Learning Stack
- **Scikit-learn 1.3.2**: Classical machine learning algorithms and preprocessing
- **XGBoost 2.0.2**: Gradient boosting framework for high-performance ML
- **Pandas 2.1.4**: Data manipulation and analysis
- **NumPy 1.24.4**: Numerical computing foundation
- **Joblib**: Model serialization and parallel processing

### Quantum Computing
- **PennyLane 0.33.1**: Quantum machine learning library for quantum circuit simulation and quantum kernels

### AI Integration
- **Google Generative AI**: Integration with Gemini API for advanced fraud pattern analysis and logical reasoning

### Data Storage
- **SQLite**: Embedded database for transaction logging and blockchain demo features (via `database.py`)
- Support for fraud transaction datasets in CSV format

### Development and Testing
- **Pytest 7.4.3**: Testing framework for unit and integration tests
- **aiohttp**: Asynchronous HTTP client for API testing
- **Requests**: HTTP library for external API calls

### Frontend Technologies
- **Bootstrap 5**: CSS framework for responsive UI design
- **Font Awesome**: Icon library for enhanced user interface
- **Custom CSS/JavaScript**: Enhanced styling and interactive features

The system is designed to be deployment-flexible, supporting both development environments with all dependencies and production environments with graceful degradation when optional components (like quantum computing libraries) are unavailable.