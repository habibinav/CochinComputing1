from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Any
from .project import (
    preprocess_text,
    validity_vectorizer,
    validity_model,
    vectorizer,
    category_model,
    priority_model,
    is_banking_query
)

app = FastAPI(
    title="Banking Query Classification System",
    version="1.0.0"
)

class ClassificationResult(BaseModel):
    query: str
    is_banking: Optional[bool] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None

class BatchRequest(BaseModel):
    queries: List[Any]  # Accept any type of input

class BatchResponse(BaseModel):
    results: List[ClassificationResult]

@app.post("/classify_batch", response_model=BatchResponse)
async def classify_batch(request: BatchRequest):
    results = []
    for raw_query in request.queries:
        try:
            # Convert non-string queries to strings
            query = str(raw_query)
            
            processed = preprocess_text(query)
            is_banking = is_banking_query(processed)
            
            if not is_banking:
                results.append(ClassificationResult(
                    query=query,
                    is_banking=False,
                    error="Non-banking query"
                ))
                continue
                
            vec = vectorizer.transform([processed])
            category = category_model.predict(vec)[0]
            priority = priority_model.predict(vec)[0]
            
            category_proba = category_model.predict_proba(vec).max()
            validity_proba = validity_model._predict_proba_lr(
                validity_vectorizer.transform([processed])
            )[0][1]
            confidence = (category_proba + validity_proba) / 2
            
            results.append(ClassificationResult(
                query=query,
                is_banking=True,
                category=category,
                priority=priority,
                confidence=float(confidence)
            ))
            
        except Exception as e:
            results.append(ClassificationResult(
                query=str(raw_query),
                error=f"Invalid enquiry: {str(e)}"
            ))
    
    return BatchResponse(results=results)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)