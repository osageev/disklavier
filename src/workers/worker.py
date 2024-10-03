class Worker:
    tag = "[red]unknwn[/red]:"

    def __init__(self, params, *, verbose: bool = False):
        self.params = params
        self.tag = params.tag
        self.verbose = verbose

    def run(self):
        raise NotImplementedError("Worker must implement run method")
