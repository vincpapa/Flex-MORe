class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_val = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_metric):
        if self.best_val is None or val_metric < self.best_val:
            self.best_val = val_metric
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")
