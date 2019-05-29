## Datasets
### 1. FB15k-237
<b> Description:</b>

The FB15K dataset was introduced in Bordes et al., 2013. It is a subset of Freebase which contains about 14,951 entities with 1,345 different relations. This dataset was found to suffer from major test leakage through inverse relations and a large number of test triples can be obtained simply by inverting triples in the training set initially by Toutanova et al.. To create a dataset without this property, Toutanova et al. introduced FB15k-237 – a subset of FB15k where inverse relations are removed.

<b>imoport:</b>

in `path/to/data` folder run:
```
$ tar -xvf WN18RR.tar.gz -C data/WN18RR
```
next step see [How to use DVC for data sets](https://gitlab.lrz.de/adl-ai/practical-big-data-science-adl-ai/wikis/How-to-use-DVC-for-data-sets)


### 2. WN18RR
<b> Description:</b>

The WN18 dataset was also introduced in Bordes et al., 2013. It included the full 18 relations scraped from WordNet for roughly 41,000 synsets. Similar to FB15K, This dataset was found to suffer from test leakage by Dettmers et al. (2018) introduced the WN18RR.

As a way to overcome this problem, Dettmers et al. (2018) introduced the WN18RR dataset, derived from WN18, which features 11 relations only, no pair of which is reciprocal (but still include four internally-symmetric relations like verb_group, allowing the rule-based system to reach 35 on all three metrics).

<b>imoport:</b>

in `path/to/data` folder run:


```
$ tar -xvf WN18RR.tar.gz -C data/WN18RR
```

next step see [How to use DVC for data sets](https://gitlab.lrz.de/adl-ai/practical-big-data-science-adl-ai/wikis/How-to-use-DVC-for-data-sets)

