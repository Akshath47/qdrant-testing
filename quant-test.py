from qdrant_client.models import ScalarQuantizationConfig

cfg = ScalarQuantizationConfig(
    type="int8",  # or ScalarType.INT8 if imported
    quantile=0.99,
    always_ram=False
)
print(cfg)
