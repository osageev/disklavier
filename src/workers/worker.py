from mido import bpm2tempo


class Worker:
    def __init__(self, params, *, bpm: int):
        self.params = params
        self.tag = params.tag
        self.verbose = params.verbose
        self.bpm = bpm
        self.tempo = bpm2tempo(self.bpm)

    def run(self):
        raise NotImplementedError("Worker must implement run method")
