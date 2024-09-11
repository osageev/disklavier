class Worker:
    tag = "[red]unknwn[/red]:"

    def __init__(self, params):
        self.params = params
        self.tag = params.tag

    def run(self):
        raise NotImplementedError("Worker must implement run method")
