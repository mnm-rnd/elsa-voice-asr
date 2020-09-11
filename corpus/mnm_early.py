from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path


def read_text(text_file):
    with open(text_file, 'r', encoding='utf-8') as out:
        return out.readlines()[0].strip()

    
class MnMAudioDataset(Dataset):
    def __init__(self, path, manifest_csv_file, tokenizer, data_transformer, bucket_size, path_from_home=True):
        if path_from_home:
            main_path = Path.home()
        else:
            main_path = Path(".")

        corpus_path = main_path.joinpath(path)
        manifest_csv_path = corpus_path.joinpath(manifest_csv_file)

        self.file_text_pair = []
        self.data_transformer = data_transformer
        self.tokenizer = tokenizer
        self.bucket_size = bucket_size
        
        with open(manifest_csv_path, 'r', encoding='utf-8') as mp:
            for x in tqdm(mp):
                str_vals = x.strip().split(",")
                
                # Preprocess the text
                text = read_text(str_vals[-1])
                text = self.data_transformer(text)
                text = self.tokenizer.encode(text)
                
                self.file_text_pair.append((str_vals[0], text))        
        
    def __len__(self):
        return len(self.file_text_pair)
    
    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self) - self.bucket_size, index)
            return self.file_text_pair[index:index+self.bucket_size]
        
        # Return a single sample
        return self.file_text_pair[index]
        
class MnMAudioTextDataset(Dataset):
    def __init__(self, path, manifest_csv_file, tokenizer, data_transformer, bucket_size, path_from_home=True):
        if path_from_home:
            main_path = Path.home()
        else:
            main_path = Path(".")

        corpus_path = main_path.joinpath(path)
        manifest_csv_path = corpus_path.joinpath(manifest_csv_file)

        self.texts = []
        self.data_transformer = data_transformer
        self.tokenizer = tokenizer
        self.bucket_size = bucket_size
        
        with open(manifest_csv_path, 'r', encoding='utf-8') as mp:
            for x in tqdm(mp):
                str_vals = x.strip().split(",")
                
                # Preprocess the text
                text = read_text(str_vals[-1])
                text = self.data_transformer(text)
                text = self.tokenizer.encode(text)
                
                self.texts.append(text)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self) - self.bucket_size, index)
            return self.texts[index:index+self.bucket_size]
        
        # Return a single sample
        return self.texts[index]
