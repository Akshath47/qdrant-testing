import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, models, ScalarQuantization, QuantizationConfig, ScalarType, ScalarQuantizationConfig
)

# 1. Set up in-memory Qdrant client
client = QdrantClient(":memory:")
collection_dim = 80
num_points = 20_000
query_vector = np.random.rand(collection_dim).tolist()

# 2. Generate random vectors
points = [
    PointStruct(
        id=i,
        vector=np.random.rand(collection_dim).tolist(),
        payload={"label": f"vec_{i}"}
    )
    for i in range(num_points)
]

# 3. Create collections
client.create_collection(
    collection_name="no_quant",
    vectors_config=VectorParams(size=collection_dim, distance=Distance.COSINE)
)

client.create_collection(
    collection_name="with_quant",
    vectors_config=VectorParams(
        size=collection_dim,
        distance=Distance.COSINE,
    ),
    quantization_config=ScalarQuantizationConfig(
        type=ScalarType.INT8,
        quantile=0.99,
        always_ram=False
    )
)

# 4. Upload points to both
client.upload_points(collection_name="no_quant", points=points)
client.upload_points(collection_name="with_quant", points=points)

# 5. Query both and measure time
def timed_query(collection_name):
    start = time.time()
    client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=5
    )
    end = time.time()
    return end - start

# Warm-up run (to stabilize any lazy loading)
_ = timed_query("no_quant")
_ = timed_query("with_quant")

# Actual timed queries
time_no_quant = timed_query("no_quant")
time_with_quant = timed_query("with_quant")

# 6. Output comparison
print(f"Query time without quantization: {time_no_quant:.6f} seconds")
print(f"Query time with scalar quantization: {time_with_quant:.6f} seconds")
print(f"Speedup: {time_no_quant / time_with_quant:.2f}x faster with quantization")
