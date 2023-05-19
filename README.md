#  A Novel Price Prediction Service for E-Commerce Categorical Data  [(Go-to-Paper)](https://www.mdpi.com/2227-7390/11/8/1938) [(Download-PDF)](https://www.mdpi.com/2227-7390/11/8/1938/pdf?version=1681986986)


## Abstract
Most e-commerce data include items that belong to different categories, e.g., product types on Amazon and eBay. The accurate prediction of an item’s price on an e-commerce platform will facilitate the maximization of economic benefits for the seller and buyer. Consequently, the task of price prediction of e-commerce items can be seen as a multiple regression on categorical data. Performing multiple regression tasks with categorical independent variables is tricky since the observations of each product type might have different distribution shapes, whereas the distribution shape of all the data might not be representative of each group. In this vein, we propose a service for facilitating the price prediction task of e-commerce categorical products. The main novelty of the proposed service relies on two unique data transformations aiming at increasing the between-group variance and decreasing the within-group variance to improve the task of regression analysis on categorical data. The proposed data transformations are tested on four different e-commerce datasets over a set of linear, non-linear, and neural network-based regression models. Comparing the best existing regression models without applying the proposed transformation, the proposed transformation results show improvements in the range of 1.98% to 8.91% for the four evaluation metrics scores, namely, R2, MAE, RMSE, and MAPE. However, the best metrics improvement on each dataset has average values of 16.8%, 8.0%, 6.0%, and 25.0% for R2, MAE, RMSE, and MAPE, respectively.

## Running an Experiment
```python
from time import time

```

## Citing

If you use the proposed simulation in your work, please cite the accompanying [paper]:

```bibtex
@article{fathalla2023novel,
  title={A Novel Price Prediction Service for E-Commerce Categorical Data},
  author={Fathalla, Ahmed and Salah, Ahmad and Ali, Ahmed},
  journal={Mathematics},
  volume={11},
  number={8},
  pages={1938},
  year={2023},
  publisher={MDPI}
}
```
[paper]: https://www.mdpi.com/1424-8220/21/17/5777
