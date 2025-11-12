import os
from getpass import getpass
from pathlib import Path

# cache_dir = input("Indicate path to all Hugging Face caches:")
cache_dir = "cache"
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
try:
    os.environ["HF_TOKEN"]
except:
    os.environ["HF_TOKEN"] = getpass(f"HuggingFace token (puis presser Entrée):\n")

from rank_comparia.pipeline import RankingPipeline

pipeline = RankingPipeline(
    method="ml",
    include_votes=True,
    include_reactions=True,
    bootstrap_samples=1000,
    mean_how="token",
    export_path=Path("output"),
)

# ranker = pipeline.ranker
scores = pipeline.run()

# Copy ml_final_data.json

# dataset_info = api.dataset_info('ministere-culture/comparia-votes')
# timestamp = getattr(dataset_info, 'lastModified', None)
# timestamp = None

# if timestamp:
#     formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
#     print(f'DATASET_TIMESTAMP={formatted_timestamp}')
# else:
#     print(f'DATASET_TIMESTAMP=unknown_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
