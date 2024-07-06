from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    """
    This class is responisble for maintaing text data as vectors.

    Attributes
    --------------------
    seq_length - int: Sequence Length
    chars - list(str): List of characters
    char_to_idx - dict: dicitonary from character to index
    idx_to_char - dict: dictionary from index to character
    vocab_size - int: Vocabolary size
    data_size - int: total length of the text
    """

    def __init__(self, text_data: str, seq_length: int = 25) -> None:
        """
        Initialisation function 

        text_data = Full text data as string
        seq_length = sequence length. 
        """ 
        self.chars = sorted(list(set(text_data)))
        self.data_size, self.vocab_size = len(text_data), len(self.chars)
        self.idx_to_char = {i:ch for i, ch in enumerate(self.chars)}
        self.char_to_idx = {ch:i for i, ch in enumerate(self.chars)}
        self.seq_length = seq_length
        self.X = self.string_to_vector(text_data)

    @property
    def X_string(self) -> str:
        """
            Returns X in string form
        """
        return self.vector_to_string(self.X)
        
    def __len__(self):
        return int(len(self.X)/self.seq_length - 1 )
    
    def __getitem__(self, index) -> tuple[torch.Tensor,torch.Tensor]:
        start_idx = index * self.seq_length
        end_idx = (index + 1) * self.seq_length

        X = torch.tensor(self.X[start_idx:end_idx]).float()
        y = torch.tensor(self.X[start_idx+1:end_idx]).float()
        return X, y
    
    def string_to_vector(self, name:str) -> list[int]:
        vector = list()
        for s in name:
            vector.append(self.char_to_idx[s])
        return vector
    
    def vector_to_string(self, vector: list[int]) -> str:
        vector_string = ""
        for i in vector:
            vector_string = ""
            for i in vector:
                vector_string += self.idx_to_char[i]
            return vector_string