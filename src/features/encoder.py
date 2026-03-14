import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class FeatureEncoder:
    def __init__(self, feature_store_dir: str):
        self.feature_store_dir = Path(feature_store_dir)
        self.feature_store_dir.mkdir(parents=True, exist_ok=True)
        
        # 类别特征的词表映射 (Vocabularies): { 'user_id': {1: 1, 2: 2...}, 'genres': {'Action': 1, ...} }
        self.vocabularies = {}
        # 连续特征的归一化器
        self.scalers = {}
        
        # 记录词表大小，用于配置 Embedding 层
        self.vocab_sizes = {}

    def fit_categorical(self, series: pd.Series, feature_name: str, is_list=False):
        """为类别特征构建词表，支持处理列表类型的特征 (如 genres)"""
        print(f"Building vocabulary for {feature_name}...")
        unique_values = set()
        
        # 0 作为 OOV / Padding 的保留索引
        vocab = {"<PAD>": 0} 
        idx = 1
        
        if is_list:
            for item_list in series.dropna():
                if isinstance(item_list, list):
                    unique_values.update(item_list)
                elif isinstance(item_list, np.ndarray):
                    unique_values.update(item_list.tolist())
        else:
            unique_values.update(series.dropna().unique())
            
        for val in sorted(list(unique_values)):
            vocab[val] = idx
            idx += 1
            
        self.vocabularies[feature_name] = vocab
        self.vocab_sizes[feature_name] = len(vocab)
        print(f"Vocab size for {feature_name}: {self.vocab_sizes[feature_name]}")

    def transform_categorical(self, series: pd.Series, feature_name: str, is_list=False, max_len=None):
        """将类别特征转换为对应的索引，如果是列表，进行 Padding"""
        vocab = self.vocabularies.get(feature_name, {"<PAD>": 0})
        
        def encode_item(val):
            return vocab.get(val, 0)

        def process_element(x):
            if is_list:
                # 处理列表：[ 'Action', 'Sci-Fi' ] -> [ 1, 5, 0, 0 ]
                if not isinstance(x, (list, np.ndarray)):
                    items = []
                else:
                    items = [encode_item(i) for i in x]
                
                if max_len:
                    if len(items) >= max_len:
                        return items[:max_len]
                    else:
                        return items + [0] * (max_len - len(items))
                return items
            else:
                # 处理单个标量
                return encode_item(x)

        return series.apply(process_element)

    def fit_continuous(self, df: pd.DataFrame, columns: list):
        """为连续特征拟合 MinMaxScaler"""
        print(f"Fitting scalers for {columns}...")
        for col in columns:
            scaler = MinMaxScaler()
            # 填充缺失值为中位数
            values = df[col].fillna(df[col].median()).values.reshape(-1, 1)
            scaler.fit(values)
            self.scalers[col] = scaler

    def transform_continuous(self, df: pd.DataFrame, columns: list):
        """转换连续特征"""
        out_df = pd.DataFrame(index=df.index)
        for col in columns:
            scaler = self.scalers[col]
            values = df[col].fillna(df[col].median()).values.reshape(-1, 1)
            out_df[col] = scaler.transform(values).flatten()
        return out_df

    def save(self):
        """保存字典和归一化器到磁盘"""
        with open(self.feature_store_dir / "vocabularies.pkl", "wb") as f:
            pickle.dump(self.vocabularies, f)
        with open(self.feature_store_dir / "vocab_sizes.pkl", "wb") as f:
            pickle.dump(self.vocab_sizes, f)
        with open(self.feature_store_dir / "scalers.pkl", "wb") as f:
            pickle.dump(self.scalers, f)
        print(f"Encoders saved to {self.feature_store_dir}")

    def load(self):
        """从磁盘加载字典和归一化器"""
        with open(self.feature_store_dir / "vocabularies.pkl", "rb") as f:
            self.vocabularies = pickle.load(f)
        with open(self.feature_store_dir / "vocab_sizes.pkl", "rb") as f:
            self.vocab_sizes = pickle.load(f)
        with open(self.feature_store_dir / "scalers.pkl", "rb") as f:
            self.scalers = pickle.load(f)
        print(f"Encoders loaded from {self.feature_store_dir}")

