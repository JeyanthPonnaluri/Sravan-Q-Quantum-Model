// Quantum Fraud Detection Web App JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize fraud score circles
    initializeFraudScoreCircles();
    
    // Initialize form validation
    initializeFormValidation();
    
    // Initialize tooltips
    initializeTooltips();
});

// Initialize fraud score circles with dynamic colors
function initializeFraudScoreCircles() {
    const circles = document.querySelectorAll('.fraud-score-circle');
    circles.forEach(circle => {
        const score = parseFloat(circle.dataset.score);
        let color;
        
        if (score < 25) {
            color = '#198754'; // Success green
        } else if (score < 50) {
            color = '#20c997'; // Teal
        } else if (score < 75) {
            color = '#ffc107'; // Warning yellow
        } else {
            color = '#dc3545'; // Danger red
        }
        
        circle.style.setProperty('--score', score);
        circle.style.background = `conic-gradient(
            ${color} 0deg,
            ${color} ${score * 3.6}deg,
            #e9ecef ${score * 3.6}deg,
            #e9ecef 360deg
        )`;
    });
}

// Form validation and enhancement
function initializeFormValidation() {
    const form = document.getElementById('transactionForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (form && analyzeBtn) {
        form.addEventListener('submit', function(e) {
            // Add loading state
            analyzeBtn.classList.add('loading');
            analyzeBtn.disabled = true;
            
            // Validate amount
            const amount = parseFloat(document.getElementById('amount').value);
            if (amount < 0) {
                e.preventDefault();
                alert('Transaction amount cannot be negative');
                analyzeBtn.classList.remove('loading');
                analyzeBtn.disabled = false;
                return;
            }
            
            if (amount > 10000000) { // 1 crore limit
                if (!confirm('This is a very large transaction (>₹1 crore). Are you sure you want to analyze it?')) {
                    e.preventDefault();
                    analyzeBtn.classList.remove('loading');
                    analyzeBtn.disabled = false;
                    return;
                }
            }
        });
        
        // Auto-update weekend flag based on day selection
        const daySelect = document.getElementById('day_of_week');
        const weekendSelect = document.getElementById('is_weekend');
        
        if (daySelect && weekendSelect) {
            daySelect.addEventListener('change', function() {
                const day = this.value;
                if (day === 'Saturday' || day === 'Sunday') {
                    weekendSelect.value = '1';
                } else {
                    weekendSelect.value = '0';
                }
            });
        }
    }
}

// Initialize Bootstrap tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Fill form with low risk example
function fillLowRisk() {
    document.getElementById('amount').value = '2500';
    document.getElementById('hour_of_day').value = '14';
    document.getElementById('is_weekend').value = '0';
    document.getElementById('day_of_week').value = 'Tuesday';
    document.getElementById('sender_age_group').value = '26-35';
    document.getElementById('receiver_age_group').value = '26-35';
    document.getElementById('sender_state').value = 'Mumbai';
    document.getElementById('sender_bank').value = 'SBI';
    document.getElementById('receiver_bank').value = 'SBI';
    document.getElementById('merchant_category').value = 'Grocery';
    document.getElementById('device_type').value = 'Android';
    document.getElementById('transaction_type').value = 'P2M';
    document.getElementById('network_type').value = '4G';
    document.getElementById('transaction_status').value = 'SUCCESS';
    
    // Scroll to form
    document.getElementById('transactionForm').scrollIntoView({ behavior: 'smooth' });
    
    // Highlight the amount field
    const amountField = document.getElementById('amount');
    amountField.focus();
    amountField.style.backgroundColor = '#e7f3ff';
    setTimeout(() => {
        amountField.style.backgroundColor = '';
    }, 2000);
}

// Fill form with high risk example
function fillHighRisk() {
    document.getElementById('amount').value = '95000';
    document.getElementById('hour_of_day').value = '3';
    document.getElementById('is_weekend').value = '1';
    document.getElementById('day_of_week').value = 'Saturday';
    document.getElementById('sender_age_group').value = '18-25';
    document.getElementById('receiver_age_group').value = '56+';
    document.getElementById('sender_state').value = 'Delhi';
    document.getElementById('sender_bank').value = 'HDFC';
    document.getElementById('receiver_bank').value = 'Unknown Bank';
    document.getElementById('merchant_category').value = 'Entertainment';
    document.getElementById('device_type').value = 'Android';
    document.getElementById('transaction_type').value = 'P2P';
    document.getElementById('network_type').value = 'WiFi';
    document.getElementById('transaction_status').value = 'SUCCESS';
    
    // Scroll to form
    document.getElementById('transactionForm').scrollIntoView({ behavior: 'smooth' });
    
    // Highlight the amount field
    const amountField = document.getElementById('amount');
    amountField.focus();
    amountField.style.backgroundColor = '#ffe6e6';
    setTimeout(() => {
        amountField.style.backgroundColor = '';
    }, 2000);
}

// Format currency input
function formatCurrency(input) {
    let value = input.value.replace(/[^\d.]/g, '');
    if (value) {
        // Add commas for thousands
        value = parseFloat(value).toLocaleString('en-IN');
        input.value = value;
    }
}

// Real-time amount formatting
document.addEventListener('DOMContentLoaded', function() {
    const amountInput = document.getElementById('amount');
    if (amountInput) {
        amountInput.addEventListener('input', function() {
            // Remove any non-numeric characters except decimal point
            this.value = this.value.replace(/[^\d.]/g, '');
            
            // Ensure only one decimal point
            const parts = this.value.split('.');
            if (parts.length > 2) {
                this.value = parts[0] + '.' + parts.slice(1).join('');
            }
            
            // Limit decimal places to 2
            if (parts[1] && parts[1].length > 2) {
                this.value = parts[0] + '.' + parts[1].substring(0, 2);
            }
        });
        
        amountInput.addEventListener('blur', function() {
            // Format the number when user leaves the field
            const value = parseFloat(this.value);
            if (!isNaN(value)) {
                this.value = value.toFixed(2);
            }
        });
    }
});

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add loading animation to buttons
function addLoadingToButton(button, text = 'Processing...') {
    button.classList.add('loading');
    button.disabled = true;
    const originalText = button.innerHTML;
    button.setAttribute('data-original-text', originalText);
    button.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>${text}`;
}

function removeLoadingFromButton(button) {
    button.classList.remove('loading');
    button.disabled = false;
    const originalText = button.getAttribute('data-original-text');
    if (originalText) {
        button.innerHTML = originalText;
    }
}

// Auto-save form data to localStorage
function saveFormData() {
    const form = document.getElementById('transactionForm');
    if (form) {
        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        localStorage.setItem('fraudDetectionFormData', JSON.stringify(data));
    }
}

// Load form data from localStorage
function loadFormData() {
    const savedData = localStorage.getItem('fraudDetectionFormData');
    if (savedData) {
        try {
            const data = JSON.parse(savedData);
            Object.keys(data).forEach(key => {
                const element = document.getElementById(key);
                if (element) {
                    element.value = data[key];
                }
            });
        } catch (e) {
            console.log('Error loading saved form data:', e);
        }
    }
}

// Save form data on input change
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('transactionForm');
    if (form) {
        // Load saved data
        loadFormData();
        
        // Save data on change
        form.addEventListener('input', saveFormData);
        form.addEventListener('change', saveFormData);
    }
});

// Clear saved form data
function clearSavedData() {
    localStorage.removeItem('fraudDetectionFormData');
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const form = document.getElementById('transactionForm');
        if (form) {
            form.submit();
        }
    }
    
    // Escape to clear form
    if (e.key === 'Escape') {
        const form = document.getElementById('transactionForm');
        if (form && confirm('Clear all form data?')) {
            form.reset();
            clearSavedData();
        }
    }
});

// Add form reset functionality
function resetForm() {
    const form = document.getElementById('transactionForm');
    if (form && confirm('Are you sure you want to reset all fields?')) {
        form.reset();
        clearSavedData();
    }
}

// Progress indicator for form completion
function updateFormProgress() {
    const form = document.getElementById('transactionForm');
    if (!form) return;
    
    const requiredFields = form.querySelectorAll('[required]');
    const filledFields = Array.from(requiredFields).filter(field => field.value.trim() !== '');
    const progress = (filledFields.length / requiredFields.length) * 100;
    
    // Update progress bar if it exists
    const progressBar = document.querySelector('.form-progress');
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
    }
}

// Monitor form completion
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('transactionForm');
    if (form) {
        form.addEventListener('input', updateFormProgress);
        form.addEventListener('change', updateFormProgress);
        updateFormProgress(); // Initial check
    }
});