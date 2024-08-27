from dataclasses import dataclass, field
from mido import Message, MetaMessage
from queue import PriorityQueue

from utils import console


@dataclass(order=True)
class MidiEvent:
    t_start: float
    event: Message | MetaMessage = field(compare=False)

    def print(self):
        return f"{self.t_start:06.02f}\t{self.midi_event}"


class MidiQueue(PriorityQueue):
    tag = "[grey60]queue[\grey60] :"

    def __init__(self, tag):
        super(PriorityQueue).__init__()
        self.tag = tag
        self.repeat_start_index = None  # Start index of the repeat section
        self.repeat_end_index = None  # End index of the repeat section

    def set_repeat_section(self, start_index: int, end_index: int) -> None:
        if start_index < 0 or end_index >= len(self.qsize()) or start_index > end_index:
            raise ValueError(
                f"Invalid repeat section indices ({start_index}, {end_index}) -> (0, {self.qsize()})"
            )
        self.repeat_start_index = start_index
        self.repeat_end_index = end_index

    def get_next_item(self, current_index, nowait=True) -> MidiEvent | None:
        if current_index < len(self.queue) - 1:
            if nowait:
                return self.get_nowait()
            else:
                return self.get()
        elif self.repeat_start_index is not None:
            return self.peek(self.repeat_start_index)
        else:
            return None  # End of queue reached with no repeat

    def peek(self, index: int) -> MidiEvent:
        """Peek at the element at the specified index in the priority queue.

        Args:
            index (int): The index of the element to peek at.

        Returns:
            The element at the specified index, if it exists.

        Raises:
            IndexError: If the index is out of range.
        """
        with self.mutex:
            if 0 <= index < len(self.queue):
                return self.queue[index]
            else:
                raise IndexError("Index out of range")

    def reset_queue(self):
        self.queue = []
        self.repeat_start_index = None
        self.repeat_end_index = None

    def print_queue(self):
        console.log(self.tag + "=" * 60)
        for msg in self.queue:
            console.log(f"{self.tag} {msg.print()}")
        console.log(self.tag + "=" * 60)
