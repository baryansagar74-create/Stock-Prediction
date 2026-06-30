// Global Variables
let priceChart = null;
let currentTicker = 'AAPL'; // Default
let autoRefreshInterval = null;

// Get Ticker from Input
function getTicker() {
    const dashInput = document.getElementById('dash-ticker');
    const homeInput = document.getElementById('ticker-input');
    
    if (dashInput && dashInput.value) return dashInput.value.toUpperCase();
    if (homeInput && homeInput.value) return homeInput.value.toUpperCase();
    
    return currentTicker;
}

// Display Status Message
function showMessage(msg, isError = false) {
    const el = document.getElementById('status-message');
    if (!el) return;
    el.innerHTML = `<div class="alert ${isError ? 'alert-danger' : 'alert-info'} py-2 mb-0">${msg}</div>`;
    setTimeout(() => { el.innerHTML = ''; }, 5000);
}

// Format Currency
const formatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
});

// Train Model
async function initiateTraining() {
    const ticker = getTicker();
    if (!ticker) return showMessage('Please enter a ticker symbol', true);
    
    showMessage(`<i class="fas fa-spinner fa-spin me-2"></i> Initiating training for ${ticker}. This may take a few minutes...`);
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker })
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            showMessage(`<i class="fas fa-check-circle me-2 text-success"></i> ${data.message}. You can continue using the app.`);
        } else {
            showMessage(data.detail || data.message, true);
        }
    } catch (error) {
        showMessage('Failed to connect to server.', true);
    }
}

// Predict Future Price
async function predictNow() {
    const ticker = getTicker();
    if (!ticker) return showMessage('Please enter a ticker symbol', true);
    
    showMessage(`<i class="fas fa-spinner fa-spin me-2"></i> Generating prediction for ${ticker}...`);
    
    // If we are on the home page, redirect to dashboard with ticker
    if (window.location.pathname === '/') {
        window.location.href = `/dashboard?ticker=${ticker}`;
        return;
    }
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker })
        });
        const data = await response.json();
        
        if (response.ok && data.status === 'success') {
            showMessage(`<i class="fas fa-check-circle me-2 text-success"></i> Prediction successful!`);
            loadDashboard(); // Refresh all data
        } else {
            showMessage(data.detail || 'Prediction failed. Model might not be trained.', true);
        }
    } catch (error) {
        showMessage('Failed to connect to server.', true);
    }
}

// Load Dashboard Data
async function loadDashboard() {
    currentTicker = getTicker();
    if (!currentTicker) return;
    
    if (document.getElementById('current-company')) {
        document.getElementById('current-company').innerText = `Dashboard: ${currentTicker}`;
    }
    
    await Promise.all([
        fetchLiveInfo(),
        fetchMetrics(),
        fetchHistory(),
        fetchTrainingLogs()
    ]);
    
    document.getElementById('last-updated').innerText = `Last Updated: ${new Date().toLocaleTimeString()}`;
}

// Fetch Live Info
async function fetchLiveInfo() {
    try {
        const response = await fetch(`/api/live/${currentTicker}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            const info = data.data;
            document.getElementById('live-price').innerText = formatter.format(info.current_price);
            document.getElementById('live-prev-close').innerText = formatter.format(info.previous_close);
            document.getElementById('live-open').innerText = formatter.format(info.open);
            document.getElementById('live-high').innerText = formatter.format(info.day_high);
            document.getElementById('live-low').innerText = formatter.format(info.day_low);
            document.getElementById('live-volume').innerText = info.volume.toLocaleString();
        }
    } catch (e) {
        console.error("Error fetching live info", e);
    }
}

// Fetch Metrics
async function fetchMetrics() {
    try {
        const response = await fetch(`/api/metrics/${currentTicker}`);
        const data = await response.json();
        
        if (data.status === 'success' && data.data) {
            const m = data.data;
            document.getElementById('metric-rmse').innerText = m.rmse.toFixed(4);
            document.getElementById('metric-mae').innerText = m.mae.toFixed(4);
            document.getElementById('metric-mape').innerText = m.mape.toFixed(2) + '%';
            document.getElementById('metric-r2').innerText = m.r2_score.toFixed(4);
            document.getElementById('metric-da').innerText = m.directional_accuracy.toFixed(2) + '%';
        }
    } catch (e) {
        console.error("Error fetching metrics", e);
    }
}

// Fetch Training Logs
async function fetchTrainingLogs() {
    try {
        const response = await fetch(`/api/training-logs/${currentTicker}`);
        const data = await response.json();
        
        const tbody = document.getElementById('training-logs-body');
        if (!tbody) return;
        
        if (data.status === 'success' && data.data.length > 0) {
            tbody.innerHTML = '';
            data.data.forEach(log => {
                tbody.innerHTML += `
                    <tr>
                        <td>${new Date(log.timestamp).toLocaleDateString()}</td>
                        <td>${log.epochs}</td>
                        <td>${log.final_loss.toFixed(4)}</td>
                        <td>${log.final_val_loss.toFixed(4)}</td>
                    </tr>
                `;
            });
        }
    } catch (e) {
        console.error("Error fetching training logs", e);
    }
}

// Fetch Prediction History & Update Charts
async function fetchHistory() {
    try {
        const response = await fetch(`/api/history/${currentTicker}`);
        const data = await response.json();
        
        if (data.status === 'success' && data.data.length > 0) {
            const history = data.data;
            
            // Update Prediction Card with latest prediction
            const latest = history[0];
            document.getElementById('pred-price').innerText = formatter.format(latest.prediction);
            
            const changeEl = document.getElementById('pred-change');
            const changeVal = latest.prediction_change_pct;
            changeEl.innerText = (changeVal > 0 ? '+' : '') + changeVal.toFixed(2) + '%';
            changeEl.className = changeVal > 0 ? 'fs-5 fw-bold text-success' : 'fs-5 fw-bold text-danger';
            
            let ciStr = "--";
            try {
                if(latest.confidence_interval) {
                    let ciArr = JSON.parse(latest.confidence_interval);
                    ciStr = `[${formatter.format(ciArr[0])}, ${formatter.format(ciArr[1])}]`;
                }
            } catch(e) {}
            document.getElementById('pred-ci').innerText = ciStr;
            document.getElementById('pred-timestamp').innerText = new Date(latest.timestamp).toLocaleString();

            // Update History Table
            const tbody = document.getElementById('history-table-body');
            tbody.innerHTML = '';
            history.forEach(row => {
                const isUp = row.prediction_change_pct > 0;
                tbody.innerHTML += `
                    <tr>
                        <td>${new Date(row.timestamp).toLocaleString()}</td>
                        <td>${row.actual_price ? formatter.format(row.actual_price) : '--'}</td>
                        <td class="text-primary fw-bold">${formatter.format(row.prediction)}</td>
                        <td class="${isUp ? 'text-success' : 'text-danger'}">
                            ${(row.prediction_change_pct > 0 ? '+' : '') + row.prediction_change_pct.toFixed(2)}%
                        </td>
                    </tr>
                `;
            });
            
            // Update Chart
            updateChart(history);
        }
    } catch (e) {
        console.error("Error fetching history", e);
    }
}

// Initialize Chart.js
function initChart() {
    const ctx = document.getElementById('priceChart');
    if (!ctx) return;
    
    Chart.defaults.color = '#5D6D7E';
    Chart.defaults.font.family = 'Lato';
    
    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Actual Price',
                    data: [],
                    borderColor: '#0A192F',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 3,
                    tension: 0.1
                },
                {
                    label: 'Predicted Price',
                    data: [],
                    borderColor: '#C5A059',
                    backgroundColor: 'rgba(197, 160, 89, 0.15)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 3,
                    tension: 0.1,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    backgroundColor: 'rgba(253, 251, 247, 0.95)',
                    titleColor: '#0A192F',
                    bodyColor: '#0A192F',
                    borderColor: '#D4C4A8',
                    borderWidth: 1
                }
            },
            scales: {
                x: { grid: { color: 'rgba(0, 0, 0, 0.05)' } },
                y: { grid: { color: 'rgba(0, 0, 0, 0.05)' } }
            }
        }
    });
}

function updateChart(history) {
    if (!priceChart) return;
    
    // Sort chronological for chart
    const sorted = [...history].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    
    const labels = sorted.map(row => new Date(row.timestamp).toLocaleTimeString());
    const actual = sorted.map(row => row.actual_price);
    const predicted = sorted.map(row => row.prediction);
    
    priceChart.data.labels = labels;
    priceChart.data.datasets[0].data = actual;
    priceChart.data.datasets[1].data = predicted;
    priceChart.update();
}

// Download Handlers
async function downloadPDF() {
    const ticker = getTicker();
    if (!ticker) return;
    
    showMessage(`Generating PDF for ${ticker}...`);
    try {
        const response = await fetch('/api/download-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${ticker}_Report.pdf`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            showMessage('');
        } else {
            showMessage('Failed to generate PDF. Make sure model is trained.', true);
        }
    } catch (e) {
        showMessage('Error downloading PDF', true);
    }
}

async function downloadCSV() {
    const ticker = getTicker();
    if (!ticker) return;
    
    try {
        const response = await fetch('/api/download-csv', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${ticker}_History.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }
    } catch (e) {
        console.error("Error downloading CSV", e);
    }
}

// Auto Refresh Logic
const toggle = document.getElementById('autoRefreshToggle');
if (toggle) {
    toggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            autoRefreshInterval = setInterval(() => {
                predictNow(); // Make a new prediction
                loadDashboard();
            }, 60000); // 1 minute
        } else {
            clearInterval(autoRefreshInterval);
        }
    });
}
