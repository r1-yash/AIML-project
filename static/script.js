document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault(); // Prevent the form from submitting normally

    const featuresInput = document.getElementById('features').value;
    const featuresArray = featuresInput.split(',').map(item => parseFloat(item.trim()));

    if (featuresArray.length !== 11) {
        alert("Please enter exactly 11 comma-separated values.");
        return;
    }

    const data = {
        features: featuresArray
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.prediction !== undefined) {
            document.getElementById('predictionResult').innerText = data.prediction;
        } else {
            document.getElementById('predictionResult').innerText = 'Error: ' + data.error;
        }
    })
    .catch(error => {
        document.getElementById('predictionResult').innerText = 'Error: ' + error.message;
    });
});
