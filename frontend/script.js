document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // UI Elements
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('span');
    const spinner = document.getElementById('loading-spinner');
    
    const gaugeFill = document.getElementById('gauge-fill');
    const probValue = document.getElementById('probability-value');
    const riskBadge = document.getElementById('risk-badge');
    const riskMessage = document.getElementById('risk-message');
    const modelVariant = document.getElementById('model-variant');

    // Set loading state
    btnText.classList.add('hidden');
    spinner.classList.remove('hidden');
    submitBtn.disabled = true;

    // Gather form data
    const formData = new FormData(e.target);
    const data = {
        Age: parseFloat(formData.get('Age')),
        Gender: formData.get('Gender'),
        Tenure: parseFloat(formData.get('Tenure')),
        Usage_Frequency: parseFloat(formData.get('Usage_Frequency')),
        Support_Calls: parseFloat(formData.get('Support_Calls')),
        Payment_Delay: parseFloat(formData.get('Payment_Delay')),
        Subscription_Type: formData.get('Subscription_Type'),
        Contract_Length: formData.get('Contract_Length'),
        Total_Spend: parseFloat(formData.get('Total_Spend')),
        Last_Interaction: parseFloat(formData.get('Last_Interaction'))
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Prediction request failed');
        }

        const result = await response.json();
        
        // Update UI with results
        const probability = (result.probability * 100).toFixed(1);
        probValue.textContent = `${probability}%`;
        
        // Update Gauge SVG (circumference is ~125.6)
        const offset = 125.6 - (125.6 * result.probability);
        gaugeFill.style.strokeDashoffset = offset;
        
        // Update Risk Badge
        riskBadge.textContent = `${result.churn_risk} Risk`;
        riskBadge.className = 'risk-badge'; // reset
        if (result.prediction === 1) {
            riskBadge.classList.add('high');
            gaugeFill.style.stroke = 'var(--danger)';
        } else {
            riskBadge.classList.add('low');
            gaugeFill.style.stroke = 'var(--success)';
        }
        
        riskMessage.textContent = result.message;
        modelVariant.textContent = result.model_variant || 'production';
        
    } catch (error) {
        console.error('Error:', error);
        riskMessage.textContent = "Error communicating with the prediction server. Please ensure the backend is running.";
        riskBadge.textContent = "Error";
        riskBadge.className = 'risk-badge high';
    } finally {
        // Reset loading state
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
        submitBtn.disabled = false;
    }
});
