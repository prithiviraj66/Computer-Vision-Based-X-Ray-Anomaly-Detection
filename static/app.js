document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const opacitySlider = document.getElementById('overlay-opacity');
    const opacityVal = document.getElementById('opacity-val');
    
    // Viewer elements
    const emptyState = document.getElementById('empty-state');
    const loaderContainer = document.getElementById('loader-container');
    const viewSideBySide = document.getElementById('view-side-by-side');
    const viewOverlayContainer = document.getElementById('view-overlay-container');
    
    // Viewer images
    const imgOrigSide = document.getElementById('img-orig-side');
    const imgHeatSide = document.getElementById('img-heat-side');
    const imgOrigBase = document.getElementById('img-orig-base');
    const imgHeatOverlay = document.getElementById('img-heat-overlay');
    
    // Visual toggles
    const btnSideBySide = document.getElementById('btn-side-by-side');
    const btnOverlay = document.getElementById('btn-overlay');
    
    // Metrics panel
    const metricsPanel = document.getElementById('metrics-panel');
    const gaugeFill = document.getElementById('gauge-fill');
    const scoreText = document.getElementById('score-text');
    const statusBadge = document.getElementById('status-badge');
    const classificationText = document.getElementById('classification-text');
    const recommendationText = document.getElementById('recommendation-text');
    
    // Highlight toggling & Samples
    const toggleHighlights = document.getElementById('toggle-highlights');
    const sampleItems = document.querySelectorAll('.sample-item');
    
    // Navigation Tabs
    const btnNavPortal = document.getElementById('btn-nav-portal');
    const btnNavMetrics = document.getElementById('btn-nav-metrics');
    const portalView = document.getElementById('portal-view');
    const metricsView = document.getElementById('metrics-view');
    
    let currentMode = 'side-by-side'; // 'side-by-side' or 'overlay'
    let currentData = null; // Holds the last prediction results
    let diagnosticHistory = [];

    // History gallery elements
    const historyGallery = document.getElementById('history-gallery');
    const emptyHistory = document.getElementById('empty-history');
    const btnClearHistory = document.getElementById('btn-clear-history');

    // Load history from localStorage
    try {
        diagnosticHistory = JSON.parse(localStorage.getItem('diagnosticHistory') || '[]');
        renderHistory();
    } catch (e) {
        console.error("Failed to load history:", e);
        diagnosticHistory = [];
    }

    function renderHistory() {
        if (!historyGallery) return;

        // Save current active state before clear
        const activeHistoryId = currentData && currentData.historyId;

        // Clear previous list, keeping empty state if no items
        const items = historyGallery.querySelectorAll('.batch-item-card');
        items.forEach(i => i.remove());

        if (diagnosticHistory.length === 0) {
            emptyHistory.classList.remove('hidden');
            btnClearHistory.classList.add('hidden');
            return;
        }

        emptyHistory.classList.add('hidden');
        btnClearHistory.classList.remove('hidden');

        diagnosticHistory.forEach(item => {
            const card = document.createElement('div');
            card.className = `batch-item-card history-item-card ${activeHistoryId === item.id ? 'active' : ''}`;
            card.setAttribute('data-id', item.id);
            
            // Format time display
            const timeStr = new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            card.innerHTML = `
                <img class="batch-item-thumb" src="${item.data.original}">
                <div class="batch-item-info">
                    <span class="batch-item-name" title="${item.name}">${item.name}</span>
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:0.25rem;">
                        <span class="batch-item-badge ${item.data.statusCode === 'abnormal' ? 'abnormal' : 'normal'}">
                            ${item.data.statusCode === 'abnormal' ? 'Fracture' : 'Normal'} (${item.data.score}%)
                        </span>
                        <span style="font-size:0.65rem; color:var(--text-muted); white-space:nowrap;">${timeStr}</span>
                    </div>
                </div>
            `;

            card.addEventListener('click', () => {
                // Remove active classes
                document.querySelectorAll('.history-item-card').forEach(c => c.classList.remove('active'));
                document.querySelectorAll('.sample-item').forEach(c => c.classList.remove('active'));
                document.querySelectorAll('.batch-item-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');

                // Load this prediction data
                loadPredictionData(item.data, item.id);
            });

            historyGallery.appendChild(card);
        });
    }

    function addToHistory(name, data) {
        // Capping history limit to 8 items to prevent localStorage size issues
        const maxHistory = 8;
        
        // Generate unique ID
        const id = Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        
        // Add to beginning of array
        diagnosticHistory.unshift({
            id: id,
            name: name,
            timestamp: new Date().toISOString(),
            data: data
        });

        // Cap array
        if (diagnosticHistory.length > maxHistory) {
            diagnosticHistory = diagnosticHistory.slice(0, maxHistory);
        }

        // Save and render
        try {
            localStorage.setItem('diagnosticHistory', JSON.stringify(diagnosticHistory));
        } catch (e) {
            console.warn("Failed to write to localStorage:", e);
        }
        renderHistory();
    }

    // Helper to load prediction data into viewport
    function loadPredictionData(data, historyId = null) {
        currentData = { ...data, historyId: historyId };

        // Populate image elements
        imgOrigSide.src = data.original;
        imgHeatSide.src = data.heatmap;
        imgOrigBase.src = data.original;
        imgHeatOverlay.src = data.heatmap;

        // Set opacity overlay value from slider
        imgHeatOverlay.style.opacity = opacitySlider.value / 100;

        // Hide loader and empty state, show panels
        emptyState.classList.add('hidden');
        loaderContainer.classList.add('hidden');
        updateViewMode();
        metricsPanel.classList.remove('hidden');

        // Render metrics and highlights
        updateMetrics(data);
        drawAnnotations(data.annotations);
    }

    // Clear history handler
    if (btnClearHistory) {
        btnClearHistory.addEventListener('click', () => {
            diagnosticHistory = [];
            try {
                localStorage.removeItem('diagnosticHistory');
            } catch (e) {
                console.warn("Failed to clear localStorage:", e);
            }
            renderHistory();
        });
    }

    // Navigation Controls
    btnNavPortal.addEventListener('click', () => {
        btnNavPortal.classList.add('active');
        btnNavMetrics.classList.remove('active');
        portalView.classList.remove('hidden');
        metricsView.classList.add('hidden');
    });
    
    btnNavMetrics.addEventListener('click', () => {
        btnNavMetrics.classList.add('active');
        btnNavPortal.classList.remove('active');
        portalView.classList.add('hidden');
        metricsView.classList.remove('hidden');
    });
    
    // Opacity control
    opacitySlider.addEventListener('input', (e) => {
        const val = e.target.value;
        opacityVal.textContent = `${val}%`;
        imgHeatOverlay.style.opacity = val / 100;
    });
    
    // Toggle highlights checkbox
    toggleHighlights.addEventListener('change', () => {
        if (currentData) {
            drawAnnotations(currentData.annotations);
        }
    });

    // Sample scan clicks
    sampleItems.forEach(item => {
        item.addEventListener('click', () => {
            const sampleId = item.getAttribute('data-sample');
            
            // Set active class
            sampleItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            handleSampleSelect(sampleId);
        });
    });
    
    // Toggle View modes
    btnSideBySide.addEventListener('click', () => {
        if (currentMode === 'side-by-side') return;
        currentMode = 'side-by-side';
        btnSideBySide.classList.add('active');
        btnOverlay.classList.remove('active');
        updateViewMode();
    });
    
    btnOverlay.addEventListener('click', () => {
        if (currentMode === 'overlay') return;
        currentMode = 'overlay';
        btnOverlay.classList.add('active');
        btnSideBySide.classList.remove('active');
        updateViewMode();
    });
    
    function updateViewMode() {
        if (!currentData) return;
        
        if (currentMode === 'side-by-side') {
            viewSideBySide.classList.remove('hidden');
            viewOverlayContainer.classList.add('hidden');
        } else {
            viewSideBySide.classList.add('hidden');
            viewOverlayContainer.classList.remove('hidden');
        }
    }
    
    // Drag and Drop event handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });
    
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 1) {
            handleBatchUpload(files);
        } else if (files.length === 1) {
            handleFileUpload(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files.length > 1) {
            handleBatchUpload(files);
        } else if (files.length === 1) {
            handleFileUpload(files[0]);
        }
    });
    
    // Click on dropzone triggers input click
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    function handleFileUpload(file) {
        // Remove active state from sample gallery items
        sampleItems.forEach(i => i.classList.remove('active'));
        
        // UI states: Show loader, hide images & empty states
        emptyState.classList.add('hidden');
        viewSideBySide.classList.add('hidden');
        viewOverlayContainer.classList.add('hidden');
        loaderContainer.classList.remove('hidden');
        metricsPanel.classList.add('hidden');
        
        // Clear previous annotations
        clearAnnotations();
        customAnnotations = [];
        if (typeof clearCustomGroup !== 'undefined' && clearCustomGroup) {
            clearCustomGroup.classList.add('hidden');
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Analysis failed.');
            }
            return response.json();
        })
        .then(data => {
            // Add to history
            addToHistory(file.name, data);
            
            // Load prediction data
            const lastItem = diagnosticHistory[0];
            loadPredictionData(data, lastItem ? lastItem.id : null);
            
            // Sync history active UI
            renderHistory();
        })
        .catch(err => {
            console.error(err);
            loaderContainer.classList.add('hidden');
            emptyState.classList.remove('hidden');
            alert('Diagnostic scan failed. Please try a different scan file.');
        });
    }
    
    function updateMetrics(data) {
        // 1. Update Anomaly score index
        scoreText.textContent = `${data.score}%`;
        
        // Radial progress animation
        const radius = 50;
        const circumference = 2 * Math.PI * radius; // 314.159
        // Map 0-100 score to gauge fill offset (0% = 314, 100% = 0)
        const offset = circumference - (data.score / 100) * circumference;
        gaugeFill.style.strokeDashoffset = offset;
        
        // Update colors based on classification
        const suggestionsList = document.getElementById('suggestions-list');
        suggestionsList.innerHTML = '';
        
        if (data.statusCode === 'abnormal') {
            gaugeFill.style.stroke = 'var(--neon-crimson)';
            statusBadge.className = 'status-badge abnormal';
            classificationText.textContent = data.classification;
            recommendationText.innerHTML = `<strong>Attention Required:</strong> Structural anomalies detected. An overlay mapping of structural deviations is highlighted in <span style="color:var(--neon-crimson); font-weight:bold;">crimson</span>. Clinical review advised.`;
        } else {
            gaugeFill.style.stroke = 'var(--neon-green)';
            statusBadge.className = 'status-badge normal';
            classificationText.textContent = data.classification;
            recommendationText.innerHTML = `<strong>Recommendation:</strong> No abnormal structures or fractures detected. The bone structure is consistent with normal distributions.`;
        }
        
        // Populate custom suggestions if available
        if (data.suggestions && data.suggestions.length > 0) {
            data.suggestions.forEach(sug => {
                const li = document.createElement('li');
                li.style.marginBottom = '0.25rem';
                li.textContent = sug;
                suggestionsList.appendChild(li);
            });
        }
    }

    function handleSampleSelect(sampleId) {
        // UI states: Show loader, hide images & empty states
        emptyState.classList.add('hidden');
        viewSideBySide.classList.add('hidden');
        viewOverlayContainer.classList.add('hidden');
        loaderContainer.classList.remove('hidden');
        metricsPanel.classList.add('hidden');
        
        // Clear previous annotations
        clearAnnotations();
        customAnnotations = [];
        if (typeof clearCustomGroup !== 'undefined' && clearCustomGroup) {
            clearCustomGroup.classList.add('hidden');
        }

        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ sample_id: sampleId })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Analysis failed.');
            }
            return response.json();
        })
        .then(data => {
            // Add to history (e.g. "Patient 019")
            const patientName = sampleId.replace('IMG', 'Patient ').replace('.jpg', '');
            addToHistory(patientName, data);
            
            // Load prediction data
            const lastItem = diagnosticHistory[0];
            loadPredictionData(data, lastItem ? lastItem.id : null);
            
            // Sync history active UI
            renderHistory();
        })
        .catch(err => {
            console.error(err);
            loaderContainer.classList.add('hidden');
            emptyState.classList.remove('hidden');
            alert('Diagnostic scan failed. Please try a different scan file.');
        });
    }

    function clearAnnotations() {
        const containers = document.querySelectorAll('.annotations-overlay');
        containers.forEach(c => c.innerHTML = '');
    }

    function drawAnnotations(annotations) {
        clearAnnotations();
        if (!annotations || !toggleHighlights.checked) return;

        const containers = document.querySelectorAll('.annotations-overlay');
        containers.forEach(c => {
            annotations.forEach(ann => {
                const circle = document.createElement('div');
                circle.className = 'fracture-circle';
                // Scale coordinates to percentages based on the 512x512 canvas size
                circle.style.left = `${(ann.cx / 512) * 100}%`;
                circle.style.top = `${(ann.cy / 512) * 100}%`;
                circle.style.width = `${(ann.r * 2 / 512) * 100}%`;
                circle.style.height = `${(ann.r * 2 / 512) * 100}%`;
                c.appendChild(circle);
            });
        });
    }

    // Close batch dashboard button click
    const btnCloseBatch = document.getElementById('btn-close-batch');
    if (btnCloseBatch) {
        btnCloseBatch.addEventListener('click', () => {
            document.getElementById('batch-dashboard').classList.add('hidden');
        });
    }

    // Batch results variable
    let batchResults = [];
    
    function handleBatchUpload(files) {
        const batchDashboard = document.getElementById('batch-dashboard');
        const progressText = document.getElementById('batch-progress-text');
        const progressPercent = document.getElementById('batch-progress-percent');
        const progressFill = document.getElementById('batch-progress-fill');
        const resultsList = document.getElementById('batch-results-list');
        
        batchDashboard.classList.remove('hidden');
        resultsList.innerHTML = '';
        batchResults = [];
        
        const total = files.length;
        let completed = 0;
        
        progressText.textContent = `Processing scans: 0 of ${total}`;
        progressPercent.textContent = `0%`;
        progressFill.style.width = `0%`;
        
        // Disable click on dropZone temporarily
        dropZone.style.pointerEvents = 'none';
        
        // Sequential upload
        function processNext(index) {
            if (index >= total) {
                // Done processing batch!
                dropZone.style.pointerEvents = 'auto';
                return;
            }
            
            const file = files[index];
            progressText.textContent = `Processing scan ${index + 1} of ${total}: ${file.name}`;
            
            const formData = new FormData();
            formData.append('file', file);
            
            fetch('/api/predict', {
                method: 'POST',
                body: formData
            })
            .then(res => {
                if (!res.ok) throw new Error("Server error");
                return res.json();
            })
            .then(data => {
                completed++;
                const pct = Math.round((completed / total) * 100);
                progressPercent.textContent = `${pct}%`;
                progressFill.style.width = `${pct}%`;
                
                // Store result
                batchResults.push({
                    name: file.name,
                    data: data
                });
                
                // Add to history
                addToHistory(file.name, data);
                
                // Create result list item card
                const card = document.createElement('div');
                card.className = 'batch-item-card';
                card.innerHTML = `
                    <img class="batch-item-thumb" src="${data.original}">
                    <div class="batch-item-info">
                        <span class="batch-item-name">${file.name}</span>
                        <span class="batch-item-badge ${data.statusCode === 'abnormal' ? 'abnormal' : 'normal'}">${data.statusCode === 'abnormal' ? 'Fracture' : 'Normal'} (Score: ${data.score}%)</span>
                    </div>
                `;
                
                // Add click handler to load this scan into main portal view
                card.addEventListener('click', () => {
                    document.querySelectorAll('.batch-item-card').forEach(c => c.classList.remove('active'));
                    card.classList.add('active');
                    
                    // Clear custom annotations on new scan selection
                    customAnnotations = [];
                    if (typeof clearCustomGroup !== 'undefined' && clearCustomGroup) {
                        clearCustomGroup.classList.add('hidden');
                    }
                    
                    // Load the result to main viewport
                    const matchingHistoryItem = diagnosticHistory.find(h => h.name === file.name);
                    loadPredictionData(data, matchingHistoryItem ? matchingHistoryItem.id : null);
                    
                    // Sync history active UI
                    renderHistory();
                });
                
                resultsList.appendChild(card);
                
                // Auto-load first item into main view
                if (index === 0) {
                    card.click();
                }
                
                processNext(index + 1);
            })
            .catch(err => {
                console.error("Batch processing error on file", file.name, err);
                completed++;
                const pct = Math.round((completed / total) * 100);
                progressPercent.textContent = `${pct}%`;
                progressFill.style.width = `${pct}%`;
                
                const card = document.createElement('div');
                card.className = 'batch-item-card';
                card.innerHTML = `
                    <div style="font-size: 1.5rem; color: var(--neon-crimson); margin-left: 0.5rem;"><i class="fa-solid fa-triangle-exclamation"></i></div>
                    <div class="batch-item-info">
                        <span class="batch-item-name">${file.name}</span>
                        <span class="batch-item-badge abnormal">Failed</span>
                    </div>
                `;
                resultsList.appendChild(card);
                processNext(index + 1);
            });
        }
        
        processNext(0);
    }

    // Interactive Drawing Mode
    let isDrawing = false;
    let startX = 0;
    let startY = 0;
    let customAnnotations = []; // holds user-drawn circles
    let tempCircle = null;
    
    const toggleDrawMode = document.getElementById('toggle-draw-mode');
    const clearCustomGroup = document.getElementById('clear-custom-group');
    const btnClearCustom = document.getElementById('btn-clear-custom');
    
    // Toggle draw mode checkbox handler
    toggleDrawMode.addEventListener('change', () => {
        const wrappers = document.querySelectorAll('.img-wrapper');
        if (toggleDrawMode.checked) {
            wrappers.forEach(w => w.classList.add('draw-mode'));
            toggleHighlights.checked = true;
        } else {
            wrappers.forEach(w => w.classList.remove('draw-mode'));
        }
    });
    
    // Clear custom highlights handler
    btnClearCustom.addEventListener('click', () => {
        customAnnotations = [];
        clearCustomGroup.classList.add('hidden');
        if (currentData) {
            drawAnnotations(currentData.annotations);
        } else {
            clearAnnotations();
        }
    });
    
    // Event delegation on annotations overlays for drawing
    const overlays = document.querySelectorAll('.annotations-overlay');
    overlays.forEach(overlay => {
        overlay.addEventListener('mousedown', (e) => {
            if (!toggleDrawMode.checked) return;
            isDrawing = true;
            
            // Get coordinates relative to the overlay container bounds
            const rect = overlay.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;
            
            // Convert startX and startY to 512-grid coordinates
            const cx = (startX / rect.width) * 512;
            const cy = (startY / rect.height) * 512;
            
            tempCircle = {
                cx: cx,
                cy: cy,
                r: 5
            };
            
            drawTempCircle(overlay);
        });
        
        overlay.addEventListener('mousemove', (e) => {
            if (!isDrawing || !tempCircle) return;
            
            const rect = overlay.getBoundingClientRect();
            const currX = e.clientX - rect.left;
            const currY = e.clientY - rect.top;
            
            // Calculate distance
            const dx = currX - startX;
            const dy = currY - startY;
            const dist = Math.sqrt(dx*dx + dy*dy);
            
            // Convert radius to 512-grid
            const r = (dist / rect.width) * 512;
            
            tempCircle.r = Math.max(5, r);
            
            drawTempCircle(overlay);
        });
        
        overlay.addEventListener('mouseup', () => {
            if (!isDrawing || !tempCircle) return;
            isDrawing = false;
            
            customAnnotations.push(tempCircle);
            tempCircle = null;
            
            clearCustomGroup.classList.remove('hidden');
            
            const tempDivs = document.querySelectorAll('.temp-draw-circle');
            tempDivs.forEach(d => d.remove());
            
            const modelAnns = currentData ? currentData.annotations : [];
            drawAnnotations(modelAnns.concat(customAnnotations));
        });
        
        overlay.addEventListener('mouseleave', () => {
            if (isDrawing) {
                isDrawing = false;
                tempCircle = null;
                const tempDivs = document.querySelectorAll('.temp-draw-circle');
                tempDivs.forEach(d => d.remove());
            }
        });
    });
    
    function drawTempCircle(overlay) {
        const existing = overlay.querySelector('.temp-draw-circle');
        if (existing) existing.remove();
        
        if (!tempCircle) return;
        
        const circle = document.createElement('div');
        circle.className = 'fracture-circle temp-draw-circle';
        circle.style.borderStyle = 'dashed';
        circle.style.left = `${(tempCircle.cx / 512) * 100}%`;
        circle.style.top = `${(tempCircle.cy / 512) * 100}%`;
        circle.style.width = `${(tempCircle.r * 2 / 512) * 100}%`;
        circle.style.height = `${(tempCircle.r * 2 / 512) * 100}%`;
        overlay.appendChild(circle);
    }

    // PDF clinical report generator
    function generatePDFReport(actionType) {
        if (!currentData) {
            alert('No diagnostic results to export.');
            return;
        }
        
        // Create elegant print container for PDF
        const printContainer = document.createElement('div');
        printContainer.id = 'pdf-report-printout';
        printContainer.style.padding = '10px 20px 20px 20px';
        printContainer.style.margin = '0px';
        printContainer.style.background = '#ffffff';
        printContainer.style.color = '#1e293b';
        printContainer.style.fontFamily = "'Outfit', sans-serif";
        printContainer.style.width = '700px';
        printContainer.style.boxSizing = 'border-box';
        
        const suggestionsListHtml = document.getElementById('suggestions-list').innerHTML;
        const recommendationsHtml = document.getElementById('recommendation-text').innerHTML;
        
        printContainer.innerHTML = `
            <div style="border-bottom: 2px solid #00f2fe; padding-bottom: 15px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0; font-size: 24px; color: #0f172a; font-weight: 700;">BoneScan<span style="color: #4facfe;">AI</span> Clinical Report</h1>
                    <p style="margin: 5px 0 0 0; font-size: 11px; color: #64748b;">Unsupervised Deep Learning Anomaly Diagnostics</p>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; font-size: 12px; font-weight: 600; color: #0f172a;">Date: ${new Date().toLocaleDateString()}</p>
                    <p style="margin: 3px 0 0 0; font-size: 10px; color: #64748b;">Report ID: BSR-${Math.floor(100000 + Math.random() * 900000)}</p>
                </div>
            </div>
            
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; margin-bottom: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px;">
                <div><strong>Patient Identity:</strong> Anonymous Demonstration</div>
                <div><strong>Diagnostic Status:</strong> <span style="font-weight: 700; color: ${currentData.statusCode === 'abnormal' ? '#ff0844' : '#38ef7d'};">${currentData.classification}</span></div>
                <div><strong>Anomaly Index:</strong> ${currentData.score}%</div>
                <div><strong>Calibrated Threshold:</strong> ${currentData.threshold}%</div>
            </div>
            
            <div style="display: flex; gap: 20px; margin-bottom: 20px; justify-content: center;">
                <div style="text-align: center; width: 300px;">
                    <h3 style="font-size: 12px; color: #64748b; text-transform: uppercase; margin-bottom: 5px;">Original X-Ray</h3>
                    <div style="width: 300px; height: 300px; border: 1px solid #cbd5e1; border-radius: 6px; overflow: hidden; background: #000; display: flex; justify-content: center; align-items: center; position: relative;">
                        <img src="${currentData.original}" style="max-width: 100%; max-height: 100%; object-fit: contain;">
                        <!-- Overlay annotations -->
                        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;">
                            ${currentData.annotations.concat(customAnnotations).map(ann => `
                                <div style="position: absolute; border: 3px solid #ff0844; border-radius: 50%; width: ${(ann.r * 2 / 512) * 100}%; height: ${(ann.r * 2 / 512) * 100}%; left: ${(ann.cx / 512) * 100}%; top: ${(ann.cy / 512) * 100}%; transform: translate(-50%, -50%); box-shadow: 0 0 8px rgba(255, 8, 68, 0.4);"></div>
                            `).join('')}
                        </div>
                    </div>
                </div>
                <div style="text-align: center; width: 300px;">
                    <h3 style="font-size: 12px; color: #64748b; text-transform: uppercase; margin-bottom: 5px;">Anomaly Heatmap</h3>
                    <div style="width: 300px; height: 300px; border: 1px solid #cbd5e1; border-radius: 6px; overflow: hidden; background: #000; display: flex; justify-content: center; align-items: center;">
                        <img src="${currentData.heatmap}" style="max-width: 100%; max-height: 100%; object-fit: contain;">
                    </div>
                </div>
            </div>
            
            <div style="border-top: 1px solid #cbd5e1; padding-top: 15px; margin-bottom: 20px;">
                <h3 style="margin: 0 0 10px 0; font-size: 14px; color: #0f172a; font-weight: 700;">Findings & Recommendations</h3>
                <div style="background: #f8fafc; border-left: 4px solid ${currentData.statusCode === 'abnormal' ? '#ff0844' : '#38ef7d'}; padding: 12px; font-size: 12px; line-height: 1.5;">
                    <p style="margin: 0;">${recommendationsHtml}</p>
                </div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <h3 style="margin: 0 0 10px 0; font-size: 14px; color: #0f172a; font-weight: 700;">Clinical Suggestions</h3>
                <ul style="margin: 0; padding-left: 20px; font-size: 12px; color: #334155; line-height: 1.6;">
                    ${suggestionsListHtml}
                </ul>
            </div>
            
            <div style="border-top: 1px solid #cbd5e1; padding-top: 15px; text-align: center; font-size: 10px; color: #64748b; margin-top: 40px;">
                <p style="margin: 0;">BoneScanAI is an unsupervised deep learning assistant. Clinical findings should be reviewed and verified by a licensed physician before finalizing diagnosis.</p>
            </div>
        `;
        
        document.body.appendChild(printContainer);
        
        const opt = {
            margin: [5, 10, 5, 10], // top, left, bottom, right in mm
            filename: `Clinical_Report_${currentData.statusCode}_${new Date().toISOString().slice(0,10)}.pdf`,
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2, useCORS: true, logging: false, scrollY: 0, scrollX: 0 },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
        };
        
        const worker = html2pdf().from(printContainer).set(opt);
        
        if (actionType === 'view') {
            worker.output('bloburl').then((blobUrl) => {
                window.open(blobUrl, '_blank');
                printContainer.remove();
            }).catch(err => {
                console.error(err);
                printContainer.remove();
                alert('Failed to display PDF.');
            });
        } else {
            worker.save().then(() => {
                printContainer.remove();
            }).catch(err => {
                console.error(err);
                printContainer.remove();
                alert('Failed to export PDF.');
            });
        }
    }

    // Attach listeners
    const btnViewPdf = document.getElementById('btn-view-pdf');
    if (btnViewPdf) {
        btnViewPdf.addEventListener('click', () => generatePDFReport('view'));
    }

    const btnExportPdf = document.getElementById('btn-export-pdf');
    if (btnExportPdf) {
        btnExportPdf.addEventListener('click', () => generatePDFReport('export'));
    }
});
