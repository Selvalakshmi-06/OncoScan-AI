const API_URL = "http://127.0.0.1:5000";

const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");
const fileName = document.getElementById("fileName");
const analyzeButton = document.getElementById("analyzeButton");

const loading = document.getElementById("loading");
const results = document.getElementById("results");


// =========================
// IMAGE SELECTION
// =========================

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    fileName.textContent = file.name;

    const imageURL = URL.createObjectURL(file);

    imagePreview.src = imageURL;

    imagePreview.style.display = "block";

    analyzeButton.disabled = false;

    results.style.display = "none";

});


// =========================
// ANALYZE MRI
// =========================

analyzeButton.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an MRI image first.");
        return;
    }


    // Show loading
    loading.style.display = "block";

    results.style.display = "none";

    analyzeButton.disabled = true;


    // Create form data
    const formData = new FormData();

    formData.append("image", file);


    try {

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",
                body: formData
            }
        );


        if (!response.ok) {

            throw new Error(
                `Server returned ${response.status}`
            );

        }


        const data = await response.json();

        console.log("API Response:", data);


        // Display prediction
        document.getElementById("prediction").textContent =
            data.prediction;

        document.getElementById("confidence").textContent =
            `${data.confidence.toFixed(2)}%`;


        // Display probabilities
        updateProbability(
            "glioma",
            data.probabilities.glioma
        );

        updateProbability(
            "meningioma",
            data.probabilities.meningioma
        );

        updateProbability(
            "notumor",
            data.probabilities.notumor
        );

        updateProbability(
            "pituitary",
            data.probabilities.pituitary
        );


        // Display Grad-CAM
        const gradcamImage =
            document.getElementById("gradcamImage");

        gradcamImage.src =
            `${API_URL}${data.gradcam_url}?t=${Date.now()}`;


        // Show results
        results.style.display = "block";


        // Scroll to results
        results.scrollIntoView({
            behavior: "smooth"
        });

    }

    catch (error) {

        console.error("Error:", error);

        alert(
            "Unable to connect to the OncoScan AI backend.\n\n" +
            "Make sure Flask is running on port 5000."
        );

    }

    finally {

        loading.style.display = "none";

        analyzeButton.disabled = false;

    }

});


// =========================
// UPDATE PROBABILITY BAR
// =========================

function updateProbability(className, value) {

    const bar =
        document.getElementById(`${className}Bar`);

    const text =
        document.getElementById(`${className}Value`);


    bar.style.width = `${value}%`;

    text.textContent =
        `${value.toFixed(2)}%`;

}