

# Requirements
python 3.7
anaconda or miniconda
https://github.com/Kaggle/kaggle-api
https://github.com/trent-b/iterative-stratification

# Dataset:
kaggle competitions download -c bengaliai-cv19

# Conda Env

    conda create --name bengali-ai python=3.7 pandas pillow
    conda install -c trent-b iterative-stratification
    conda install -c conda-forge pyarrow tqdm imgaug
    conda install albumentations -c albumentations
    conda install pytorch torchvision cpuonly -c pytorch

# Reference
https://www.youtube.com/watch?v=8J5Q4mEzRtY



# Steps
1. `mkdir input` and `pip install --user kaggle` and add path using `export PATH="$HOME/.local/bin:$PATH"` and add kaggle credential using https://github.com/Kaggle/kaggle-api/blob/master/README.md 
2. Download dataset in `input` folder using `kaggle competitions download -c bengaliai-cv19` command
3. unzip it using `unzip bengaliai-cv19.zip`
4. run `create_folds.py`
5. check the .parquet data file using `check_dataframes.py`, if data is readable then conintue
6. Create .pickle files of the dataset using `create_image_pickles.py`. This is because training will be faster with pickles
7. 