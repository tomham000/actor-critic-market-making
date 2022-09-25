from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record('wealth', self.model.get_env().get_attr('wealth')[0])
        self.logger.record('position', self.model.get_env().get_attr('position')[0])
        return True
