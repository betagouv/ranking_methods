from datetime import datetime
from huggingface_hub import HfApi
import os

try:
    api = HfApi()

    dataset_info = api.dataset_info('ministere-culture/comparia-votes')
    timestamp = getattr(dataset_info, 'lastModified', None)
    timestamp = None
    
    if timestamp:
        formatted_timestamp = timestamp.strftime('%Y%m%d_%H%M%S')
        print(f'DATASET_TIMESTAMP={formatted_timestamp}')
    else:
        print(f'DATASET_TIMESTAMP=unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}')
        
except Exception as e:
    print(f'Error: {e}')
    print(f'DATASET_TIMESTAMP=error_{datetime.now().strftime('%Y%m%d_%H%M%S')}')