// JavaScript for fraud detection form handling and result display

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('fraudForm');
    const resultCard = document.getElementById('resultCard');
    const loading = document.querySelector('.loading');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading, hide results
        loading.style.display = 'block';
        resultCard.style.display = 'none';
        
        // Collect form data
        const formData = new FormData(form);
        const data = {
            amount: parseFloat(formData.get('amount')),
            sender_age_group: formData.get('sender_age_group'),
            merchant_category: formData.get('merchant_category'),
            hour_of_day: parseInt(formData.get('hour_of_day')),
            is_weekend: parseInt(formData.get('is_weekend')),
            device_type: formData.get('device_type')
        };
        
        // Add logical score if provided
        const pLogic = formData.get('p_logic');
        if (pLogic && pLogic.trim() !== '') {
            data.p_logic = parseFloat(pLogic);
        }
        
        try {
            // Make API request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            // Display results
            displayResults(result);
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error analyzing transaction: ' + error.message);
        } finally {
            // Hide loading
            loading.style.display = 'none';
        }
    });
});

function displayResults(result) {
    // Update progress bars and scores
    updateScore('quantum', result.quantum_score);
    updateScore('classical', result.classical_score);
    updateScore('logical', result.logical_score);
    updateScore('practical', result.practical_score);
    
    // Update uncertainty
    document.getElementById('uncertainty').textContent = result.uncertainty.toFixed(2);
    
    // Update risk assessment
    updateRiskAssessment(result.practical_score, result.uncertainty);
    
    // Show result card
    document.getElementById('resultCard').style.display = 'block';
    
    // Scroll to results
    document.getElementById('resultCard').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start' 
    });
}

function updateScore(type, score) {
    const progressBar = document.getElementById(type + 'Progress');
    const scoreText = document.getElementById(type + 'Score');
    
    // Update progress bar
    progressBar.style.width = score + '%';
    progressBar.setAttribute('aria-valuenow', score);
    
    // Update score text with color coding
    scoreText.textContent = score.toFixed(2) + '%';
    
    // Color coding based on score
    let colorClass = 'score-low';
    if (score >= 70) {
        colorClass = 'score-high';
        progressBar.className = 'progress-bar bg-danger';
    } else if (score >= 40) {
        colorClass = 'score-medium';
        progressBar.className = 'progress-bar bg-warning';
    } else {
        colorClass = 'score-low';
        progressBar.className = 'progress-bar bg-success';
    }
    
    scoreText.className = 'text-muted ' + colorClass;
}

function updateRiskAssessment(practicalScore, uncertainty) {
    const riskAlert = document.getElementById('riskAlert');
    const riskLevel = document.getElementById('riskLevel');
    const riskDescription = document.getElementById('riskDescription');
    
    let alertClass, level, description;
    
    if (practicalScore >= 80) {
        alertClass = 'alert-danger';
        level = '🚨 HIGH RISK';
        description = 'This transaction shows strong indicators of fraud. Immediate review recommended.';
    } else if (practicalScore >= 60) {
        alertClass = 'alert-warning';
        level = '⚠️ MEDIUM RISK';
        description = 'This transaction has elevated fraud indicators. Additional verification suggested.';
    } else if (practicalScore >= 30) {
        alertClass = 'alert-info';
        level = '🔍 LOW-MEDIUM RISK';
        description = 'This transaction shows some fraud indicators but appears mostly legitimate.';
    } else {
        alertClass = 'alert-success';
        level = '✅ LOW RISK';
        description = 'This transaction appears legitimate with minimal fraud indicators.';
    }
    
    // Add uncertainty consideration
    if (uncertainty > 10) {
        description += ' Note: High uncertainty in quantum prediction.';
    }
    
    riskAlert.className = 'alert ' + alertClass;
    riskLevel.textContent = level;
    riskDescription.textContent = description;
}

// Add some interactive features
document.addEventListener('DOMContentLoaded', function() {
    // Auto-update hour based on current time
    const hourInput = document.getElementById('hour_of_day');
    const currentHour = new Date().getHours();
    hourInput.value = currentHour;
    
    // Auto-update weekend based on current day
    const weekendSelect = document.getElementById('is_weekend');
    const currentDay = new Date().getDay();
    const isWeekend = currentDay === 0 || currentDay === 6 ? 1 : 0;
    weekendSelect.value = isWeekend;
    
    // Add tooltips for better UX
    const tooltips = {
        'amount': 'Transaction amount in Indian Rupees',
        'sender_age_group': 'Age group of the person sending money',
        'merchant_category': 'Type of merchant or transaction category',
        'hour_of_day': 'Hour when transaction occurred (0-23)',
        'is_weekend': 'Whether transaction occurred on weekend',
        'device_type': 'Device used for the transaction',
        'p_logic': 'Optional external logical/rule-based fraud score'
    };
    
    Object.keys(tooltips).forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.title = tooltips[id];
        }
    });
});