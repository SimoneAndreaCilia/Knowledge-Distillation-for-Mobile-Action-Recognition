# Data Folder

Place your datasets here. Remember that these files (especially large ones) should not be uploaded to GitHub. The `.gitignore` is set to ignore contents within this directory.

## HMDB51 Dataset

To run the experiments or perform inference, you must download the HMDB51 dataset locally.

1. Download the RAR file containing the videos from the [hugging face website](https://huggingface.co/datasets/jili5044/hmdb51).
2. Extract the contents so that the 51 action class folders (e.g., `brush_hair`, `cartwheel`) are placed directly inside `data/hmdb51/`.
3. The training and testing split annotations should be placed inside `data/hmdb51_splits/`.

Your directory structure should look exactly like this:

```
data/
├── hmdb51/
│   ├── brush_hair/
│   │   ├── video1.avi
│   │   └── ...
│   └── ... (50 other class folders)
├── hmdb51_splits/
│   ├── brush_hair_test_split1.txt
│   └── ...
```
