document.addEventListener('DOMContentLoaded', () => {
    
    const fetchBtn = document.getElementById('fetch-tx-btn');
    const fetchRandomBtn = document.getElementById('fetch-random-btn');
    const manualTxInput = document.getElementById('manual-tx-id');
    const analyzeBtn = document.getElementById('analyze-btn');
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const resultBox = document.getElementById('result-box');
    const closeResultBtn = document.getElementById('close-result');
    
    // Status polling
    const statusText = document.getElementById('system-status');
    const pulse = document.querySelector('.pulse');
    
    // Fetch info display
    const txInfoContainer = document.getElementById('tx-info-container');
    const txIdDisplay = document.getElementById('tx-id-display');
    const actualClassDisplay = document.getElementById('actual-class-display');
    
    let currentFeatures = {};

    function checkStatus() {
        fetch('/status')
            .then(res => res.json())
            .then(data => {
                statusText.textContent = data.status;
                if(data.status === "Ready") {
                    pulse.classList.add('ready');
                    pulse.classList.remove('error');
                    
                    // Enable fetch only when ready
                    fetchBtn.disabled = false;
                    fetchRandomBtn.disabled = false;
                } else if(data.status === "Error") {
                    pulse.classList.add('error');
                    pulse.classList.remove('ready');
                } else {
                    pulse.classList.remove('ready', 'error');
                    fetchBtn.disabled = true;
                    fetchRandomBtn.disabled = true;
                }
                
                if(data.status !== "Ready" && data.status !== "Error") {
                    setTimeout(checkStatus, 2000);
                }
            })
            .catch(err => console.error("Could not fetch status"));
    }
    checkStatus();

    // Fetch Transaction
    async function fetchTransaction(random = false) {
        let url = '/get_transaction';
        if (!random) {
            const txId = manualTxInput.value.trim();
            if (!txId) {
                alert("Please enter a Transaction ID, or click Random.");
                return;
            }
            url += `?id=${txId}`;
        }
        
        fetchBtn.disabled = true;
        fetchRandomBtn.disabled = true;
        
        try {
            const res = await fetch(url);
            const data = await res.json();
            
            if(!res.ok) throw new Error(data.error || "Failed to fetch transaction");
            
            // Populate UI
            txInfoContainer.classList.remove('hidden');
            txIdDisplay.textContent = data.id;
            actualClassDisplay.textContent = data.actual_class === 1 ? 'FRAUD (1)' : 'LEGITIMATE (0)';
            actualClassDisplay.style.color = data.actual_class === 1 ? 'var(--danger)' : 'var(--success)';
            
            currentFeatures = data.features;
            
            // Fill input boxes
            Object.keys(currentFeatures).forEach(key => {
                const el = document.getElementById(key);
                if (el) {
                    el.value = Number(currentFeatures[key]).toFixed(4);
                }
            });
            
            analyzeBtn.disabled = false;
            resultContainer.classList.add('hidden'); // Hide old results
            
        } catch (e) {
            alert(e.message);
        } finally {
            fetchBtn.disabled = false;
            fetchRandomBtn.disabled = false;
        }
    }

    fetchBtn.addEventListener('click', () => fetchTransaction(false));
    fetchRandomBtn.addEventListener('click', () => {
        manualTxInput.value = ''; // clear input
        fetchTransaction(true);
    });

    // Radio card logic
    const radioCards = document.querySelectorAll('.radio-card');
    radioCards.forEach(card => {
        card.querySelector('input').addEventListener('change', (e) => {
            radioCards.forEach(c => c.classList.remove('selected'));
            if(e.target.checked) card.classList.add('selected');
        });
    });

    closeResultBtn.addEventListener('click', () => {
        resultContainer.classList.add('hidden');
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI State: Loading
        const btnText = analyzeBtn.querySelector('span');
        const spinner = analyzeBtn.querySelector('.spinner');
        
        btnText.textContent = 'Analyzing Features...';
        spinner.classList.remove('loader-hidden');
        analyzeBtn.disabled = true;
        resultContainer.classList.add('hidden');

        // Form selection
        const selectedModel = document.querySelector('input[name="model"]:checked').value;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: selectedModel,
                    features: currentFeatures
                })
            });

            const result = await response.json();
            
            setTimeout(() => {
                displayResult(result);
                btnText.textContent = 'Analyze Transaction';
                spinner.classList.add('loader-hidden');
                analyzeBtn.disabled = false;
            }, 800);

        } catch (error) {
            console.error("Error:", error);
            btnText.textContent = 'Analyze Transaction';
            spinner.classList.add('loader-hidden');
            analyzeBtn.disabled = false;
            alert("Error connecting to inference server.");
        }
    });

    function displayResult(result) {
        if(result.error) {
            alert(result.error);
            return;
        }

        resultContainer.classList.remove('hidden');
        
        const probRing = document.getElementById('probability-ring');
        const probValue = document.getElementById('prob-value');
        const classBadge = document.getElementById('classification-badge');
        const insightText = document.getElementById('insight-text');

        const finalProb = Math.round(result.probability * 100);
        const circumference = 283;
        const offset = circumference - (finalProb / 100) * circumference;
        
        probRing.style.strokeDashoffset = circumference;
        setTimeout(() => { probRing.style.strokeDashoffset = offset; }, 100);

        let count = 0;
        probValue.textContent = "0%";
        const interval = setInterval(() => {
            if (count >= finalProb) {
                clearInterval(interval);
                probValue.textContent = finalProb + "%";
            } else {
                let increment = Math.ceil((finalProb - count) / 10) || 1;
                count += increment;
                if(count > finalProb) count = finalProb;
                probValue.textContent = count + "%";
            }
        }, 20);

        if (result.prediction === 1) {
            classBadge.textContent = "FRAUDULENT";
            classBadge.className = "classification fraud";
            probRing.style.stroke = "var(--danger)";
            insightText.textContent = "High alert: Detected anomalous pattern based on the 10 extracted features. Matches known fraud characteristics.";
        } else {
            classBadge.textContent = "LEGITIMATE";
            classBadge.className = "classification legit";
            probRing.style.stroke = "var(--success)";
            insightText.textContent = "The 10 extracted features align with standard user behavior. No significant anomalies detected.";
        }
    }
});
