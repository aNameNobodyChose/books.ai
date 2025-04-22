from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from quote_caster.quote_caster_inference import predict_speakers

app = FastAPI(title="QuoteCaster Inference API", version="1.0")

@app.post("/predict")
async def predict(request: Request):
    try:
        input_data = await request.json()
        if not isinstance(input_data, list):
            return JSONResponse(content={"error": "Expected a list of quote objects"}, status_code=400)

        results = predict_speakers(input_data)
        return {"results": results}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
