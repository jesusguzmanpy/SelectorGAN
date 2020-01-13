
def find_dataset_using_name(dataset_name):
    dataset = None    
    return dataset


def create_dataset(opt):

    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        
    def load_data(self):
        return self