
import Pyro4
Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED = {'pickle'}
import socket
from datasets.pyro_dataset import PyroServerDataset

from bin.experiments.semantic_segmentation.verse2019.dataset import Dataset

@Pyro4.expose
class VerseServerDataset(PyroServerDataset):
    def __init__(self):
        super(VerseServerDataset, self).__init__(queue_size=8, refill_queue_factor=0.0, n_threads=8, use_multiprocessing=False)

    def init_with_parameters(self, *args, **kwargs):
        # TODO: adapt base folder, in case this script runs on a remote server
        #kwargs['base_folder'] = 'ADAPT_TO_CORRECT_BASE_FOLDER'
        self.dataset_class = Dataset(*args, **kwargs)
        self.dataset = self.dataset_class.dataset_train()


if __name__ == '__main__':
    print('start')
    daemon = Pyro4.Daemon(host=socket.gethostname(), port=51232)
    print(daemon.register(VerseServerDataset(), 'verse_dataset'))
    daemon.requestLoop()
