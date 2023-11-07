#! /bin/bash
python -B semi_main.py --label_ratio 0.05
python -B semi_main.py --label_ratio 0.10
python -B semi_main.py --label_ratio 0.30
python -B semi_main.py --label_ratio 0.70
python -B semi_main.py --label_ratio 0.90


