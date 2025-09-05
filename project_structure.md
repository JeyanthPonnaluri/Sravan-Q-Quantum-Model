# Proposed Project Structure

## Current Issues:
- Multiple duplicate API files (simple_api.py, enhanced_api.py, demo_api.py, quantum_fraud_api.py, meta_fraud_api.py, quantum_web_app.py)
- Multiple README files for different versions
- Multiple requirements files
- Scattered model files
- Mixed test files in root and tests/ folder

## Proposed Clean Structure:

```
neuro-qkad/
├── README.md                    # Main project documentation
├── requirements.txt             # Main dependencies
├── config.yaml                  # Configuration
├── .gitignore                   # Git ignore rules
│
├── src/                         # Core source code
│   ├── __init__.py
│   ├── api_main.py             # Main FastAPI application
│   ├── data_prep.py            # Data preprocessing
│   ├── qkernel.py              # Quantum kernel implementation
│   ├── train.py                # Training pipeline
│   ├── save_load.py            # Model persistence
│   └── web/                    # Web interface
│       ├── static/
│       │   ├── style.css
│       │   └── script.js
│       └── templates/
│           ├── base.html
│           ├── index.html
│           ├── results.html
│           └── error.html
│
├── models/                     # Trained model artifacts
│   └── .gitkeep
│
├── data/                       # Data files
│   └── .gitkeep
│
├── tests/                      # All test files
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_kernel.py
│   └── test_integration.py
│
├── docs/                       # Documentation
│   ├── API_SPECIFICATION.md
│   ├── DEPLOYMENT_GUIDE.md
│   ├── QUANTUM_API_DOCS.md
│   └── QUICK_START.md
│
└── scripts/                    # Utility scripts
    ├── run_web_app.py
    └── start_quantum_api.py
```

## Files to Keep (Main Version):
- src/api_main.py (most complete)
- src/ folder structure (already good)
- Main README.md (most comprehensive)
- requirements.txt (main one)
- config.yaml
- tests/ folder content

## Files to Remove:
- All duplicate API files
- All duplicate README files  
- All duplicate requirements files
- Scattered test files in root
- Demo and simple versions