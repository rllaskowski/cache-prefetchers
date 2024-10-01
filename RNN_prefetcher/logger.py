import logging


class FileLogger(logging.Logger):
    def __init__(self, name, level=logging.INFO, filename=None):
        super().__init__(name, level)
        if filename is not None:
            self.file_handler = logging.FileHandler(filename)
            self.file_handler.setLevel(level)
            self.formatter = logging.Formatter("%(asctime)s,%(name)s,%(levelname)s,%(message)s")
            self.file_handler.setFormatter(self.formatter)
            self.addHandler(self.file_handler)

    def log_training_loss(self, epoch, loss):
        message = f"Training,{epoch},{loss:.4f}"
        self.info(message)

    def log_validation_loss(self, epoch, loss):
        message = f"Validation,{epoch},{loss:.4f}"
        self.info(message)

    def log_accuracy(self, epoch, accuracy):
        message = f"Accuracy,{epoch},{accuracy:.4f}"
        self.info(message)
