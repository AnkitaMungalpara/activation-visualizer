from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from activations import relu, leaky_relu, sigmoid, tanh, gelu, layer_output
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Activation Function Explorer")

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can set your frontend origin instead of "*" for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class ActivationRequest(BaseModel):
    x: List[float]
    function: str
    alpha: Optional[float] = 0.01
    weights: Optional[List[List[float]]] = None
    bias: Optional[List[float]] = None

@app.post("/compute")
def compute_activation(req: ActivationRequest):
    x = req.x
    fn_name = req.function.lower()
    
    fn_map = {
        "relu": relu,
        "leaky_relu": lambda x: leaky_relu(x, req.alpha),
        "sigmoid": sigmoid,
        "tanh": tanh,
        "gelu": gelu
    }
    
    if fn_name not in fn_map:
        return {"error": f"Function {req.function} not supported."}
    
    # Optional: simulate layer
    if req.weights or req.bias:
        result = layer_output(x, req.weights, req.bias, activation_fn=fn_map[fn_name])
    else:
        result = fn_map[fn_name](x)
    
    return {"input": x, "function": fn_name, "output": result}
