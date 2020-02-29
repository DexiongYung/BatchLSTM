from torch.utils.data import Dataset


class NameDataset(Dataset):
    def __init__(self, df, col_name):
        """
        Args:
            csv_file (string): Path to the csv file WITHOUT labels
            col_name (string): The column name corresponding to the people names that'll be standardized
        """
        self.data_frame = df[col_name]
        self.max_len = None

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        return self.data_frame.iloc[index]
