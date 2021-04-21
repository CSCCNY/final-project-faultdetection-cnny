import glob
import os
import pdb
import sys

DATA_FOLDER = "nepali_news_dataset_20_categories_large"

def load_data(data_folder=DATA_FOLDER):
  sys.path.append('../data')
  data = []
  full_path = f"../data/{data_folder}/**/*.txt"
  print(f"Loading file from: {full_path}")
  for categorized_article in glob.iglob(full_path, recursive=True): 
      category = categorized_article.split("/")[-2]
      # data[category] = data.get(category, [])
      with open(categorized_article, encoding="utf8", errors='ignore') as file:
        article = file.read()
        data.append([category, article])
  return data


