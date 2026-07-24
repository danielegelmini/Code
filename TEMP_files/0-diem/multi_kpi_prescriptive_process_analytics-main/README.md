# Balancing KPIs through Multi-Objective Prescriptive Process Analytics

This repository contains the official implementation of the paper:
> **Balancing KPIs through Multi-Objective Prescriptive Process Analytics**  
> *Ngoc-Diem Le, Andrei Buliga, Massimiliano Ronzani, Alessandro Padella, Massimiliano de Leoni*  
> [[Paper (PDF)](https://link.springer.com/chapter/10.1007/978-3-032-28110-4_6)] 

## Abstract
Prescriptive Process Analytics aims to support ongoing process executions by guiding them toward desirable outcomes, which are usually measured through Key Performance Indicators (KPIs). While the literature on Prescriptive Process Analytics has focused on the exploration of different recommendation techniques and tasks, the current efforts have so far largely focused on optimizing a single KPI. However, real-world processes often involve multiple, interrelated, and potentially conflicting KPIs, requiring an explicit treatment of their trade-offs. This paper presents a framework that generates recommendations designed to optimize multiple KPIs simultaneously, and that allows users to explore different levels of trade-off among them. Our proposed framework jointly recommends the next activity and its executing resource, which we call the next step recommendation, aiming to balance multiple KPIs. Different strategies are proposed to generate these recommendations, including one that utilizes a genetic algorithm to efficiently explore the search space of viable options. This is particularly effective in domains with a large number of resources, which lead to a significantly extensive space of possible recommendations. Experimental results demonstrate that our multi-KPI approach achieves favorable trade-offs among partially conflicting goals, while minimizing the degradation of each individual KPI compared to single-objective optimization.


## Framework
<img width="4572" height="2955" alt="framework_updated" src="https://github.com/user-attachments/assets/4ae419bd-22c1-444c-9ce6-44e4a1405b77" />

## Installation
**Dependencies**

This implementation requires the following Python libraries:
```python
pip install pandas numpy catboost dice_ml
```

## Usage
1. Preprocessing Event Logs
2. Training the Utility Function for multi-KPI
3. Generating Recommendations
```python
python run_experiment.py --case_study "bpi12" --method "genetic" --num_cfes 5 --window_size 5 --reduced_threshold 0.05
```

## License
This project is licensed under the MIT License.

## Citation
```bibtex
@inproceedings{le2026balancing,
  title={Balancing KPIs Through Multi-objective Prescriptive Process Analytics},
  author={Le, Ngoc-Diem and Buliga, Andrei and Ronzani, Massimiliano and Padella, Alessandro and de Leoni, Massimiliano},
  booktitle={International Conference on Advanced Information Systems Engineering},
  pages={98--115},
  year={2026},
  organization={Springer}
}
