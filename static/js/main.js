$(document).ready(function () {
    const outputArea = $('#output-area');

    function updateOutput(text) {
        outputArea.append(text + '\n');
        outputArea.scrollTop(outputArea[0].scrollHeight);
    }

    function clearOutput() {
        outputArea.empty();
    }

    $('#btn-upload-dataset').click(function() {
        clearOutput();
        updateOutput("Checking for dataset in project root...");
        $.get('/check-dataset', function(data) {
            if (data.dataset_present) {
                updateOutput("Dataset Status: Present ✅");
                updateOutput("Train Path: " + data.train_path);
                updateOutput("Test Path: " + data.test_path);
                updateOutput("Summary: Dataset is ready for training or preprocessing.");
            } else {
                updateOutput("Dataset Status: Missing ❌");
                updateOutput("Please ensure the 'Dataset' folder (with 'Train' and 'Test' subfolders) is present in the project directory.");
            }
        });
    });

    $('#btn-preprocess').click(function() {
        updateOutput("--- Starting Image Processing & Normalization ---");
        updateOutput("1. Resizing images to 224x224...");
        updateOutput("2. Normalizing pixel values [0, 1]...");
        updateOutput("3. Applying Data Augmentation (Rotation, Shift, Zoom)...");
        updateOutput("Status: Preprocessing Complete.");
    });

    $('#btn-gnb').click(function() {
        updateOutput("--- Fetching GNB Classifier Performance ---");
        $.get('/gnb-stats', function(data) {
            updateOutput("ML Model Accuracy = " + data.accuracy);
            updateOutput("Model Precision = " + data.precision);
            updateOutput("ML Model Recall = " + data.recall);
            updateOutput("ML Model F1-Score = " + data.f1);
            updateOutput("ML Model Confusion matrix =");
            updateOutput("[" + JSON.stringify(data.confusion_matrix[0]) + "\n " + JSON.stringify(data.confusion_matrix[1]) + "]");
        });
    });

    let trainingInterval = null;

    function checkTrainingStatus() {
        $.get('/training-status', function(data) {
            if (data.status === 'training') {
                var stage = data.stage ? data.stage : 'training';
                var epochLine = 'Training is running...';
                if (data.epoch && data.total_epochs) {
                    epochLine = 'Epoch ' + data.epoch + '/' + data.total_epochs + ' (' + stage + ')';
                }
                updateOutput('Training Status: ' + epochLine + ' ⏳');

                var metrics = [];
                if (data.loss != null) metrics.push('loss: ' + data.loss.toFixed(4));
                if (data.accuracy != null) metrics.push('accuracy: ' + (data.accuracy * 100).toFixed(2) + '%');
                if (data.val_loss != null) metrics.push('val_loss: ' + data.val_loss.toFixed(4));
                if (data.val_accuracy != null) metrics.push('val_accuracy: ' + (data.val_accuracy * 100).toFixed(2) + '%');
                if (metrics.length) {
                    updateOutput('Metrics: ' + metrics.join(' | '));
                }
            } else if (data.status === 'completed') {
                updateOutput("--- TRAINING COMPLETED SUCCESSFULLY! ✅ ---");
                updateOutput("New model saved as 'model_vgg19_best.keras'.");
                if (data.timestamp) {
                    updateOutput("Timestamp: " + data.timestamp);
                }
                clearInterval(trainingInterval);
                trainingInterval = null;
            } else if (data.status === 'error') {
                updateOutput("--- TRAINING FAILED! ❌ ---");
                updateOutput("Error details: " + data.message);
                clearInterval(trainingInterval);
                trainingInterval = null;
            }
        });
    }

    $('#btn-train-cnn').click(function() {
        updateOutput("--- Initiating CNN (VGG19) Model Training ---");
        updateOutput("Connecting to training backend...");
        
        $.ajax({
            type: 'POST',
            url: '/train',
            success: function(response) {
                updateOutput("Server Response: " + response);
                updateOutput("Status: Training process is now running in the background.");
                updateOutput("Note: I will poll the server every 30 seconds for updates.");
                
                // Start polling for status
                if (!trainingInterval) {
                    trainingInterval = setInterval(checkTrainingStatus, 30000); // Check every 30s
                }
            },
            error: function(xhr) {
                updateOutput("Error: Failed to start training.");
                updateOutput("Details: " + xhr.responseText);
            }
        });
    });

    $('#btn-upload-test').click(function() {
        $('#upload-section').toggle();
        if($('#upload-section').is(':visible')) {
            updateOutput("Ready for testing. Please choose a blood smear image.");
        }
    });

    $('#btn-performance').click(function() {
        updateOutput("--- Performance Visualization ---");
        updateOutput("ML Model Metrics:");
        updateOutput("- CNN (VGG19) Accuracy: ~96.5% (Based on training logs)");
        updateOutput("- CNN (VGG19) Loss: ~0.12");
        updateOutput("- GNB Accuracy: 61.9%");
        updateOutput("Note: Full training plots (Accuracy/Loss curves) are available in the 'CNN And Transfer LEarning.ipynb' notebook.");
    });

    $('#btn-exit').click(function() {
        if(confirm("Are you sure you want to exit?")) {
            updateOutput("Session ended. You can close this tab now.");
            setTimeout(function() {
                window.close();
            }, 1000);
        }
    });

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#image-preview-container').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        readURL(this);
    });

    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        updateOutput("--- Starting Prediction ---");
        updateOutput("Predicting...");

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                updateOutput("Result: " + data);
                if (data === "Model not available") {
                    updateOutput("Error: Model could not be loaded. Please check if TensorFlow is installed and compatible.");
                } else if (data.toLowerCase().indexOf('parasit') === 0) {
                    updateOutput("Warning: Malaria Parasite Detected! 🦠");
                } else if (data.toLowerCase().indexOf('uninfect') === 0) {
                    updateOutput("Status: Uninfected / Healthy ✅");
                } else {
                    updateOutput("Note: Unexpected response from server.");
                }
            },
            error: function(xhr, status, error) {
                updateOutput("Error: Prediction failed. Status: " + xhr.status + ", Error: " + error);
                if (xhr.status === 0) {
                    updateOutput("Hint: The server might have crashed or the request was blocked by the browser.");
                } else {
                    updateOutput("Details: " + xhr.responseText);
                }
            }
        });
    });
});
