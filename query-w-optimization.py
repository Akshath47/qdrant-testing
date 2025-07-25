import numpy as np
import time
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScalarQuantizationConfig,
    ScalarType,
)

# --- 1. Setup Qdrant client ---
client = QdrantClient(":memory:")

vector_dim = 128
num_vectors = 10000
query_vector = np.random.rand(vector_dim).tolist()

# --- 2. Create quantized collection ---
client.create_collection(
    collection_name="quantized_vectors",
    vectors_config=VectorParams(
        size=vector_dim,
        distance=Distance.EUCLID),
    quantization_config=ScalarQuantizationConfig(
        type=ScalarType.INT8,
        quantile=0.99,
        always_ram=False
    )
)

# --- 3. Insert vectors ---
points = [
    PointStruct(
        id=i,
        vector=np.random.rand(vector_dim).tolist()
    ) for i in range(num_vectors)
]

client.upload_points(collection_name="quantized_vectors", points=points)

# --- 4. Perform ANN query with oversampling (top 50) ---
top_k = 10
oversample_k = 50

start = time.time()
approx_results = client.query_points(
    collection_name="quantized_vectors",
    query=query_vector,
    limit=oversample_k,
    with_vectors=True
)
end = time.time()
print(f"Initial quantized ANN query took: {end - start:.5f} sec")

# --- 5. Reranking with exact L2 distance ---
def l2_dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Extract vectors for rescoring
scored = [
    (pt.id, pt.vector, l2_dist(query_vector, pt.vector))
    for pt in approx_results.points
]

# Sort by true distance
rescored = sorted(scored, key=lambda x: x[2])[:top_k]

# --- 6. Print final reranked results ---
print(f"\nTop {top_k} results after reranking:")
for i, (id_, vec, dist) in enumerate(rescored, start=1):
    print(f"{i}. ID: {id_}, True L2 Distance: {dist:.5f}")
