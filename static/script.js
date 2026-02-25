document.getElementById("predictionForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const formData = new FormData(this);

    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const html = await response.text();

    const parser = new DOMParser();
    const doc = parser.parseFromString(html, "text/html");

    const result = doc.querySelector("#resultText")?.innerText;
    const probability = doc.querySelector("#probabilityText")?.innerText;

    if (result) {
        document.getElementById("resultText").innerText = result;
        document.getElementById("probabilityText").innerText = probability;
        document.getElementById("resultBox").classList.remove("hidden");
    }
});
