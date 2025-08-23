function toggleAlpha() {
    const fn = document.getElementById("function").value;
    document.getElementById("alphaDiv").style.display = fn === "leaky_relu" ? "block" : "none";
}

function parseMatrix(str) {
    if (!str) return null;
    return str.split(';').map(row => row.split(',').map(Number));
}

function parseArray(str) {
    if (!str) return null;
    return str.split(',').map(Number);
}

async function plotActivation() {
    const fn = document.getElementById("function").value;
    const minX = parseFloat(document.getElementById("minX").value);
    const maxX = parseFloat(document.getElementById("maxX").value);
    const alpha = parseFloat(document.getElementById("alpha").value);
    const weights = parseMatrix(document.getElementById("weights").value);
    const bias = parseArray(document.getElementById("bias").value);

    // Generate inputs
    const x = [];
    for (let i = minX; i <= maxX; i += 0.1) x.push(parseFloat(i.toFixed(2)));

    // Prepare request
    const payload = { x: x, function: fn };
    if (fn === "leaky_relu") payload.alpha = alpha;
    if (weights) payload.weights = weights;
    if (bias) payload.bias = bias;

    // Call FastAPI
    const response = await fetch("http://127.0.0.1:8000/compute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await response.json();

    // Plot
    const trace = { x: x, y: data.output, mode: 'lines', name: fn };
    const layout = { title: `Activation Function: ${fn}`, xaxis: {title:'Input'}, yaxis:{title:'Output'} };
    Plotly.newPlot("plot", [trace], layout);
}
