import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

template = PROJECT_ROOT / "feast_st" / "feature_store.yaml.template"
config = PROJECT_ROOT / "feast_st" / "feature_store.yaml"

password = os.environ["FEAST_POSTGRES_PASSWORD"]

content = template.read_text()

content = content.replace(
    "__FEAST_PASSWORD__",
    password
)

config.write_text(content)

print("Feast configuration created.")