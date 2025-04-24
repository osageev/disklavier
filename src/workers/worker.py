import types
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

    def reset(self):
        """
        reset the worker's state generically.

        this method resets the base worker tempo. it also attempts to reset
        instance attributes back to their default values if they are defined
        at the class level with simple types (int, float, bool, str, list, dict, set).

        note: this generic reset will not correctly reset complex state like
        file handles, loaded data structures (e.g., pandas dataframes, faiss indexes),
        or network connections. these require specific logic, typically by overriding
        this method in the subclass and calling super().reset().
        """
        # reset worker-specific state
        self.tempo = bpm2tempo(self.bpm)  # recompute based on current bpm

        cls = type(self)
        # iterate over instance attributes that might need resetting
        for attr_name in list(self.__dict__.keys()):
            # skip config/base attributes handled elsewhere or not needing reset
            if attr_name in ["params", "tag", "verbose", "bpm", "tempo"]:
                continue
            # skip private/protected attributes by convention
            if attr_name.startswith("_"):
                continue

            # check if this attribute has a corresponding class-level attribute
            if hasattr(cls, attr_name):
                default_value = getattr(cls, attr_name)

                # skip resetting if the class attribute is a method, module, etc.
                if callable(default_value) or isinstance(
                    default_value, types.ModuleType
                ):
                    continue

                # perform reset for simple types based on the class default
                # create new empty collections for mutable types
                if isinstance(default_value, list):
                    setattr(self, attr_name, [])
                elif isinstance(default_value, dict):
                    setattr(self, attr_name, {})
                elif isinstance(default_value, set):
                    setattr(self, attr_name, set())
                elif isinstance(default_value, (int, float, bool, str)):
                    setattr(self, attr_name, default_value)
                # add other simple types here if needed
                # note: complex types are intentionally skipped

        # use console logging if available and verbose
        if self.verbose:
            try:
                from utils import console  # dynamically import if possible

                console.log(f"{self.tag} generic worker state reset performed")
            except ImportError:
                print(
                    f"{self.tag if hasattr(self, 'tag') else 'Worker'}: generic worker state reset performed"
                )
