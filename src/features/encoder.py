import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class FeatureEncoder:
    def __init__(self, feature_store_dir: str):
        self.feature_store_dir = Path(feature_store_dir)
        self.feature_store_dir.mkdir(parents=True, exist_ok=True)
        
        # 类别特征的词表映射
        self.vocabularies = {}
        # 连续特征的归一化器
        self.scalers = {}
        # 连续特征的中位数 (用于 transform 时填充 NaN)
        self.medians = {}
        # 记录词表大小
        self.vocab_sizes = {}

    def fit_categorical(self, series: pd.Series, feature_name: str, is_list=False):
        """为类别特征构建词表"""
        print(f"Building vocabulary for {feature_name}...")
        unique_values = set()
        vocab = {"<PAD>": 0} 
        idx = 1
        
        if is_list:
            for item_list in series.dropna():
                if isinstance(item_list, (list, np.ndarray)):
                    unique_values.update(item_list)
        else:
            unique_values.update(series.dropna().unique())
            
        for val in sorted(list(unique_values)):
            vocab[val] = idx
            idx += 1
            
        self.vocabularies[feature_name] = vocab
        self.vocab_sizes[feature_name] = len(vocab)
        print(f"Vocab size for {feature_name}: {self.vocab_sizes[feature_name]}")

    def transform_categorical(self, series: pd.Series, feature_name: str, is_list=False, max_len=None):
        """将类别特征转换为索引"""
        vocab = self.vocabularies.get(feature_name, {"<PAD>": 0})
        
        def encode_item(val):
            return vocab.get(val, 0)

        def process_element(x):
            if is_list:
                if not isinstance(x, (list, np.ndarray)):
                    items = []
                else:
                    items = [encode_item(i) for i in x]
                if max_len:
                    return items[:max_len] + [0] * (max_len - len(items[:max_len]))
                return items
            else:
                return encode_item(x)

        return series.apply(process_element)

    def fit_continuous(self, df: pd.DataFrame, columns: list, prefix=""):
        """为连续特征拟合 Scaler，使用 prefix 防止 key 冲突"""
        for col in columns:
            key = f"{prefix}_{col}" if prefix else col
            print(f"Fitting scaler for {key}...")
            scaler = MinMaxScaler()
            median_val = df[col].median()
            self.medians[key] = median_val
            values = df[col].fillna(median_val).values.reshape(-1, 1)
            scaler.fit(values)
            self.scalers[key] = scaler

    def transform_continuous(self, df: pd.DataFrame, columns: list, prefix=""):
        """转换连续特征"""
        out_df = pd.DataFrame(index=df.index)
        for col in columns:
            key = f"{prefix}_{col}" if prefix else col
            scaler = self.scalers[key]
            median_val = self.medians.get(key, 0.0)
            values = df[col].fillna(median_val).values.reshape(-1, 1)
            out_df[col] = scaler.transform(values).flatten()
        return out_df

    def save(self):
        artifacts = {
            'vocabularies': self.vocabularies,
            'vocab_sizes': self.vocab_sizes,
            'scalers': self.scalers,
            'medians': self.medians
        }
        with open(self.feature_store_dir / "encoder_artifacts.pkl", "wb") as f:
            pickle.dump(artifacts, f)
        print(f"Encoders saved to {self.feature_store_dir}")

    def load(self):
        with open(self.feature_store_dir / "encoder_artifacts.pkl", "rb") as f:
            artifacts = pickle.load(f)
        self.vocabularies = artifacts['vocabularies']
        self.vocab_sizes = artifacts['vocab_sizes']
        self.scalers = artifacts['scalers']
        self.medians = artifacts.get('medians', {})
        print(f"Encoders loaded from {self.feature_store_dir}")
