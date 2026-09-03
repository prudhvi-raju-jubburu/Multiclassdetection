/* ==========================================================================
   URBAN OBJECT DETECTION - FRONTEND INTERACTION LOGIC
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const dropzone = document.getElementById("dropzone");
    const dropzoneContent = document.getElementById("dropzoneContent");
    const imageInput = document.getElementById("imageInput");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const btnRemove = document.getElementById("btnRemove");

    const confSlider = document.getElementById("confSlider");
    const confValue = document.getElementById("confValue");

    const btnPredict = document.getElementById("btnPredict");
    const btnSpinner = document.getElementById("btnSpinner");
    const btnText = btnPredict.querySelector(".btn-text");

    const samplesGrid = document.getElementById("samplesGrid");

    const placeholderState = document.getElementById("placeholderState");
    const outputContainer = document.getElementById("outputContainer");
    const outputImage = document.getElementById("outputImage");
    const viewToggles = document.getElementById("viewToggles");

    const analysisDashboard = document.getElementById("analysisDashboard");
    const totalDetections = document.getElementById("totalDetections");
    const uniqueClassesCount = document.getElementById("uniqueClassesCount");
    const classBadges = document.getElementById("classBadges");
    const detectionsTableBody = document.getElementById("detectionsTableBody");

    // State Variables
    let selectedFile = null;
    let selectedSamplePath = null;
    let originalDataUrl = null;
    let annotatedDataUrl = null;

    // Class Colors Map (matches backend & UI)
    const CLASS_COLORS = {
        "person": "#ff3b30",
        "rider": "#ff9500",
        "car": "#34c759",
        "bus": "#00c7be",
        "truck": "#30b0ff",
        "bike": "#5856d6",
        "motor": "#af52de",
        "traffic light": "#ffcc00",
        "traffic sign": "#ff2d55",
        "train": "#a2845e"
    };

    // --------------------------------------------------------------------------
    // 1. SLIDER EVENT
    // --------------------------------------------------------------------------
    confSlider.addEventListener("input", (e) => {
        confValue.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // --------------------------------------------------------------------------
    // 2. DRAG & DROP & FILE SELECTION
    // --------------------------------------------------------------------------
    dropzone.addEventListener("click", (e) => {
        if (e.target.closest("#btnRemove")) return;
        imageInput.click();
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    imageInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    btnRemove.addEventListener("click", (e) => {
        e.stopPropagation();
        resetInputState();
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file (JPG, PNG, WEBP).");
            return;
        }

        selectedFile = file;
        selectedSamplePath = null;

        const reader = new FileReader();
        reader.onload = (e) => {
            originalDataUrl = e.target.result;
            imagePreview.src = originalDataUrl;
            dropzoneContent.classList.add("hidden");
            previewContainer.classList.remove("hidden");
            btnPredict.disabled = false;
            
            // Clear active sample selection highlight
            document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
        };
        reader.readAsDataURL(file);
    }

    function resetInputState() {
        selectedFile = null;
        selectedSamplePath = null;
        originalDataUrl = null;
        imageInput.value = "";
        imagePreview.src = "";
        previewContainer.classList.add("hidden");
        dropzoneContent.classList.remove("hidden");
        btnPredict.disabled = true;
        document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
    }

    // --------------------------------------------------------------------------
    // 3. QUICK SAMPLES LOADER
    // --------------------------------------------------------------------------
    async function loadSamples() {
        try {
            const resp = await fetch("/api/samples");
            const data = await resp.json();

            samplesGrid.innerHTML = "";
            if (data.samples && data.samples.length > 0) {
                data.samples.forEach(sample => {
                    const card = document.createElement("div");
                    card.className = "sample-card";
                    card.innerHTML = `<img src="/${sample.path}" alt="${sample.name}">`;
                    card.addEventListener("click", () => {
                        selectSample(sample.path, card);
                    });
                    samplesGrid.appendChild(card);
                });
            } else {
                samplesGrid.innerHTML = `<p style="font-size:12px; color:var(--text-dim); grid-column:span 3;">No sample images found</p>`;
            }
        } catch (err) {
            console.error("Failed to load sample images:", err);
            samplesGrid.innerHTML = `<p style="font-size:12px; color:var(--text-dim); grid-column:span 3;">Error loading samples</p>`;
        }
    }

    function selectSample(path, cardEl) {
        selectedFile = null;
        selectedSamplePath = path;
        originalDataUrl = `/${path}`;

        imagePreview.src = originalDataUrl;
        dropzoneContent.classList.add("hidden");
        previewContainer.classList.remove("hidden");
        btnPredict.disabled = false;

        document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
        cardEl.classList.add("active");
    }

    loadSamples();

    // --------------------------------------------------------------------------
    // 4. PREDICTION ACTION
    // --------------------------------------------------------------------------
    btnPredict.addEventListener("click", async () => {
        if (!selectedFile && !selectedSamplePath) return;

        // UI Loading State
        btnPredict.disabled = true;
        btnText.classList.add("hidden");
        btnSpinner.classList.remove("hidden");

        const formData = new FormData();
        formData.append("confidence", confSlider.value);

        if (selectedFile) {
            formData.append("image", selectedFile);
        } else if (selectedSamplePath) {
            formData.append("sample_path", selectedSamplePath);
        }

        try {
            const resp = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const result = await resp.json();

            if (resp.ok && result.success) {
                annotatedDataUrl = result.image;
                displayResults(result);
            } else {
                alert(`Detection Error: ${result.error || "Failed to process image"}`);
            }

        } catch (err) {
            console.error("Prediction error:", err);
            alert("An error occurred while connecting to the detection server.");
        } finally {
            btnPredict.disabled = false;
            btnText.classList.remove("hidden");
            btnSpinner.classList.add("hidden");
        }
    });

    // --------------------------------------------------------------------------
    // 5. RENDER RESULTS
    // --------------------------------------------------------------------------
    function displayResults(data) {
        placeholderState.classList.add("hidden");
        outputContainer.classList.remove("hidden");
        analysisDashboard.classList.remove("hidden");

        // Set output image
        outputImage.src = data.image;

        // Update toggle state to "annotated"
        document.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
        document.querySelector('.toggle-btn[data-view="annotated"]').classList.add("active");

        // Metrics
        totalDetections.textContent = data.total_detections;
        const classes = Object.keys(data.class_counts || {});
        uniqueClassesCount.textContent = classes.length;

        // Badges
        classBadges.innerHTML = "";
        if (classes.length === 0) {
            classBadges.innerHTML = `<span style="font-size:12px; color:var(--text-muted)">No objects detected above threshold</span>`;
        } else {
            classes.forEach(cls => {
                const count = data.class_counts[cls];
                const color = CLASS_COLORS[cls.toLowerCase()] || "#6366f1";
                const badge = document.createElement("span");
                badge.className = "class-count-badge";
                badge.style.borderColor = `${color}66`;
                badge.style.color = color;
                badge.innerHTML = `
                    <span class="badge-dot" style="width:8px; height:8px; border-radius:50%; background:${color}"></span>
                    ${cls}
                    <span class="badge-count-num">${count}</span>
                `;
                classBadges.appendChild(badge);
            });
        }

        // Table
        detectionsTableBody.innerHTML = "";
        if (data.detections && data.detections.length > 0) {
            data.detections.forEach((det, idx) => {
                const tr = document.createElement("tr");
                const color = CLASS_COLORS[det.class.toLowerCase()] || "#6366f1";
                const pct = (det.confidence * 100).toFixed(1);
                
                tr.innerHTML = `
                    <td>${idx + 1}</td>
                    <td><strong style="color:${color}">${det.class}</strong></td>
                    <td><span class="conf-pill">${pct}%</span></td>
                    <td><code style="font-size:11px; color:var(--text-muted)">[${det.box.join(", ")}]</code></td>
                `;
                detectionsTableBody.appendChild(tr);
            });
        } else {
            detectionsTableBody.innerHTML = `<tr><td colspan="4" style="text-align:center; color:var(--text-muted); padding:20px;">No predictions recorded</td></tr>`;
        }
    }

    // --------------------------------------------------------------------------
    // 6. TOGGLE BETWEEN ANNOTATED AND ORIGINAL IMAGE
    // --------------------------------------------------------------------------
    viewToggles.addEventListener("click", (e) => {
        const btn = e.target.closest(".toggle-btn");
        if (!btn) return;

        const view = btn.dataset.view;
        document.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        if (view === "original" && originalDataUrl) {
            outputImage.src = originalDataUrl;
        } else if (view === "annotated" && annotatedDataUrl) {
            outputImage.src = annotatedDataUrl;
        }
    });
});
