# Data anonymization using k-Anonymity
## ✔️ Experiments
- Provides 5 k-anonymization method: 
  - Datafly
  - Incognito 
  - Topdown Greedy
  - Classic Mondrian
  - Basic Mondrian
- Implements 3 anonymization metrics: 
  - Equivalent Class size metric (CAVG)
  - Discernibility Metric (DM)
  - Normalized Certainty Penalty (NCP)
- Implements 3 classification models: 
  - Random Forests 
  - Support Vector Machines 
  - K-Nearest Neighbors

## 🌟 Executing
To anonymize dataset, run:
```
from butanone.Anonymizer import Anonymizer
a = Anonymizer(model_type, k, dataset_name)
a.anonymize()
```
- **model_type**: [mondrian | classic_mondrian | mondrian_ldiv | topdown | cluster | datafly]
- **dataset_name**: [adult | cahousing | cmc | mgm | informs | italia]

Results will be in ```results/{dataset}/{method}``` folder

To run evaluation metrics on every combination of algorithms, datasets and value k, run:
```
python visualize.py
```

Results will be in ```demo/{metrics.png, metrics_ml.png}``` 

## References:
- Basic Mondrian, Top-Down Greedy, Cluster-based (https://github.com/fhstp/k-AnonML)
- L-Diversity (https://github.com/Nuclearstar/K-Anonymity, https://github.com/qiyuangong/Mondrian_L_Diversity)
- Classic Mondrian (https://github.com/qiyuangong/Mondrian)
- Datafly Algorithm (https://github.com/nazilkbahar/python-datafly)
- Normalized Certainty Penalty from [Utility-Based Anonymization for Privacy Preservation with
Less Information Loss](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.450.6140&rep=rep1&type=pdf)
- Discernibility, Average Equivalent Class Size from [A Systematic Comparison and Evaluation
of k-Anonymization Algorithms
for Practitioners](http://www.tdp.cat/issues11/tdp.a169a14.pdf)
- [Privacy in a Mobile-Social World](https://courses.cs.duke.edu//fall12/compsci590.3/slides/lec3.pdf)
- Code and idea based on [k-Anonymity in Practice: How Generalisation and Suppression Affect Machine Learning Classifiers](https://arxiv.org/abs/2102.04763)
