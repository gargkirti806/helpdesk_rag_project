from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.models.api import QueryRequest
from app.pipeline.graph import workflow
import uuid
import numpy as np
import torch

router = APIRouter()

def convert_to_json_serializable(obj):
    """
    Recursively convert objects to JSON-serializable Python types
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(v) for v in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list)):
        return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif hasattr(obj, "__dict__"):
        return convert_to_json_serializable(vars(obj))
    else:
        return obj

@router.post("/helpdesk", response_class=JSONResponse)
def handle_helpdesk(req: QueryRequest):
    """
    Handle helpdesk queries using RAG pipeline.
    
    Returns JSON response with query results.
    """
    try:
        thread_id = req.thread_id or str(uuid.uuid4())
        checkpoint_ns = req.checkpoint_ns or "helpdesk_ns"
        checkpoint_id = req.checkpoint_id or str(uuid.uuid4())

        config = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id
            }
        }

        state_input = {"user_query": req.query}
        final_state = workflow.invoke(state_input, config=config)

        # Convert final state to JSON-safe types
        safe_state = convert_to_json_serializable(final_state)

        final_result = (
            safe_state.get("final_answer") or 
            safe_state.get("response") or 
            safe_state.get("result") or 
            safe_state
        )

        response_data = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "result": final_result,
            "full_state": safe_state
        }

        # Use JSONResponse to bypass Pydantic serialization
        return JSONResponse(content=response_data)

    except Exception as e:
        import traceback
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"Error in helpdesk endpoint: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))
