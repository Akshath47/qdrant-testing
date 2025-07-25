from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. Create an in-memory Qdrant client
client = QdrantClient(":memory:")

# 2. Create a collection (e.g., 3D vectors using cosine similarity)
client.create_collection(
    collection_name="test_vectors",
    vectors_config=VectorParams(size=3, distance=Distance.COSINE)
)

# 3. Define and upload a few example vectors
points = [
    PointStruct(id=1, vector=[0.1, 0.2, 0.3], payload={"label": "A"}),
    PointStruct(id=2, vector=[0.4, 0.5, 0.6], payload={"label": "B"}),
    PointStruct(id=3, vector=[0.9, 0.0, 0.1], payload={"label": "C"})
]

client.upload_points(collection_name="test_vectors", points=points)

# 4. Query with a new vector
query_vector = [0.1, 0.2, 0.25]

results = client.query_points(
    collection_name="test_vectors",
    query=query_vector,
    limit=2
)

# 5. Print results
for hit in results.points:
    print(f"ID: {hit.id}, Score: {hit.score:.4f}, Payload: {hit.payload}")

