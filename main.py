from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from activations import relu, leaky_relu, sigmoid, tanh, gelu, layer_output

app = FastAPI()

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ComputeRequest(BaseModel):
    x: List[float]
    fn_name: str
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

@app.post("/compute")
def compute_activation(req: ComputeRequest):
    x = req.x
    fn_name = req.fn_name.lower()
    
    if fn_name not in fn_map:
        return {"error": f"Activation {fn_name} not supported."}
    
    # Update alpha if provided
    if fn_name == "leaky_relu" and req.alpha is not None:
        fn_map["leaky_relu"] = lambda x: leaky_relu(x, alpha=req.alpha)
    
    try:
        output = layer_output(
            x=x,
            weights=req.weights,
            bias=req.bias,
            activation_fn=fn_map[fn_name]
        )
    except ValueError as e:
        return {"error": str(e)}
    
    return {"output": output.tolist()}
