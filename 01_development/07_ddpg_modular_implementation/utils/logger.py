import logging

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class Logger():
    
    def __init__(self, log_dir=''):
        """
        General logger
        """
        self.writer = SummaryWriter(log_dir)
        self.info = logger.info
        self.debug = logger.debug
        self.warning = logger.warning

    def scalar_summary(self, tag, value, step):
        """
        Log scalar value
        """
        self.writer.add_scalar(tag, value, step)

    def graph_model(self, model, input):
        """
        Graph the model
        """
        self.writer.add_graph(model, input)
