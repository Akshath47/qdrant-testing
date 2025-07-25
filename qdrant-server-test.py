from qdrant_client import QdrantClient

client = QdrantClient(host="ec2-44-192-110-143.compute-1.amazonaws.com", port=6333)

print(client.get_collections())