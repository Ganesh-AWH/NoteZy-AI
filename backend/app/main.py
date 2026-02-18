from utils.utils_pipeline import UtilsServicePipeline
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

pipeline = UtilsServicePipeline()

# Ingest document
result = pipeline.ingest(r"backend\app\trw.pdf")
print(result)


# Ask questions
results = pipeline.search("Engineering ethics and Design failure?", k=3)

for r in results:
    print(f"\nScore: {r['score']:.4f}")
    print(r["text"])
