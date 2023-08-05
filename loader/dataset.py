import torch
from torch.utils.data import Dataset, DataLoader
from gensim.utils import simple_preprocess


class SentimentDataset(Dataset):
  def __init__(self, df, tokenizer, max_len =120):
    self.df = df
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):

    row = self.df.iloc[index]
    text, label = self.get_input_data(row)
    # Encode_plus will:
        # (1) split text into token
        # (2) Add the '[CLS]' and '[SEP]' token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map token to their IDS
        # (5) Create attention mask
        # (6) Return a dictionary of outputs
    encoding = self.tokenizer.encode_plus(
        text, 
        truncation= True,
        add_special_tokens = True,
        max_length = self.max_len, 
        padding = 'max_length',
        return_attention_mask = True,
        return_token_type_ids = False,
        return_tensors = 'pt'
    )
    return {
        'text':text,
        'input_ids':encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'targets': torch.tensor(label, dtype= torch.long)
    }
  def get_input_data(self, row):
    #preprocessing: {remove icon, special charater, lower}
    text = row['content']
    text = ' '.join(simple_preprocess(text))
    label = row['label']

    return text, label