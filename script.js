async function plotActivation() {
    const fn = document.getElementById("function").value;
    const minX = parseInt(document.getElementById("minX").value);
    const maxX = parseInt(document.getElementById("maxX").value);

    // Generate inputs
    const x = [];
    for (let i = minX; i <= maxX; i += 0.1) {
        x.push(parseFloat(i.toFixed(2)));
    }

    // Call FastAPI backend
    const response = await fetch("http://127.0.0.1:8000/compute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x: x, function: fn })
    });

    const data = await response.json();
    const y = data.output;

    // Plot using Plotly
    const trace = {
        x: x,
        y: y,
        mode: 'lines',
        name: fn
    };

    const layout = {
        title: `Activation Function: ${fn}`,
        xaxis: { title: "Input" },
        yaxis: { title: "Output" }
    };

    Plotly.newPlot("plot", [trace], layout);
}
