#!/usr/bin/env bash
set -euo pipefail

python -m ui_rec.src.data.generate_mock
python -m ui_rec.src.features.build_features
python -m ui_rec.src.models.exposure        
python -m ui_rec.src.models.ui_type        
python -m ui_rec.src.models.service_cluster 
python -m ui_rec.src.models.rank           
python -m ui_rec.src.inference.predict --allowed "card,list_item,banner,icon"
