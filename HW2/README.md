# CS760 Homework 2

Scripts and dependencies for generating the required decision trees

### Installation & Setup

```bash
python -m venv venv
python pip install -r requirements.txt
```

### Basic Usage

```
# ./HW2/train_dtree.py:

from dep_dtree import import_train_display_tree, validate_sample_against_model

if __name__ == '__main__':
    datasets = ('HW2/data/Druns.txt', 'HW2/data/D1.txt', 'HW2/data/D2.txt')
    for dataset in datasets:
        data_dict = import_train_display_tree(dataset)
        model, df = data_dict['model'], data_dict['data']
        validate_sample_against_model(model,df)

```

### Tree Output
```
# output from ./HW2/Druns.txt:

root
├── x_2 >= 8 (y = 1)
└── x_2 < 8
    ├── x_2 >= 0
    │   ├── x_2 >= 6
    │   │   ├── x_2 >= 7 (y = 0)
    │   │   └── x_2 < 7 (y = 1)
    │   └── x_2 < 6 (y = 0)
    └── x_2 < 0
        ├── x_1 >= 0.1 (y = 0)
        └── x_1 < 0.1 (y = 1)

branches: 5 nodes & 6 leaves
```
Reach out with any questions, thank you!

Ante TC