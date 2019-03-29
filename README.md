# SearchEngine

## How to re-build dataset
The data-set is too large and exceeds GitHub's limit of 50 MB file size limit.

I have split them using `split -b 10m dataset.csv`

To re-build the dataset, run this:

```bash
cat dataset/* > dataset/dataset.csv
```
