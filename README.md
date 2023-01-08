# READ ME

QIS stands for Quantitative investment strategies. The package implements analytics for analysis,
simulation, and visualization of quantitative strategies 

## **Installation**

```python 
pip install qis
```

## **Analytics**

The QIS package is split into subpackages based on the scope level.

The inclusion level is from a low dependency to higher dependency subpackages:


1. utils is low level utilities for pandas and numpy operations

2. perfstats is subpackage for performance statistics related to returns, volatilities, etc.

3. plots is subpackage for plotting apis

4. models includes several modules for analytical models split by applications

5. portfolio is a high level package for analysis, simulation, and reporting of quant strategies

6. data is a stand-alone package for data fetching using external apis

7. example is modul with examples of QIS analytics


