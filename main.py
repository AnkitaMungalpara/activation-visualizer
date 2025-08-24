from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from activations import relu, leaky_relu, sigmoid, tanh, gelu, layer_output
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Enable CORS for frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model (changed fn_name -> function)
class ComputeRequest(BaseModel):
    x: List[float]
    function: str   # <-- matches frontend payload
    weights: Optional[List[List[float]]] = None
    bias: Optional[List[float]] = None
    alpha: Optional[float] = 0.01  # for Leaky ReLU

# Map names to activation functions
fn_map = {
    "relu": relu,
    "leaky_relu": lambda x: leaky_relu(x, alpha=0.01),  # default alpha
    "sigmoid": sigmoid,
    "tanh": tanh,
    "gelu": gelu,
}

# Serve index.html at root
@app.get("/")
def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

@app.post("/compute")
def compute_activation(req: ComputeRequest):
    print(f"Received request: {req}")
    print(f"Request dict: {req.dict()}")
    
    fn_name = req.function.lower()  # Only need this once
    
    if fn_name not in fn_map:
        return {"error": f"Activation {fn_name} not supported."}
    
    # Update alpha if provided
    if fn_name == "leaky_relu" and req.alpha is not None:
        fn_map["leaky_relu"] = lambda x: leaky_relu(x, alpha=req.alpha)
    
    try:
        output = layer_output(
            x=req.x,
            weights=req.weights,
            bias=req.bias,
            activation_fn=fn_map[fn_name]
        )
    except ValueError as e:
        return {"error": str(e)}
    
    return {"output": output.tolist()}