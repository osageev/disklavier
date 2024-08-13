from typing import List


class QueueItem:
    def __init__(self, midi_file, start_time):
        self.midi_file = midi_file  # Reference to a PreloadedMidiFile object
        self.start_time = start_time  # Global start time for this MIDI file


class MidiQueue:
    def __init__(self):
        self.queue: List[QueueItem] = []
        self.repeat_start_index = None  # Start index of the repeat section
        self.repeat_end_index = None  # End index of the repeat section

    def add_to_queue(self, midi_file, start_time):
        queue_item = QueueItem(midi_file, start_time)
        self.queue.append(queue_item)

    def set_repeat_section(self, start_index, end_index):
        if start_index < 0 or end_index >= len(self.queue) or start_index > end_index:
            raise ValueError("Invalid repeat section indices")
        self.repeat_start_index = start_index
        self.repeat_end_index = end_index

    def get_next_item(self, current_index):
        if current_index < len(self.queue) - 1:
            return self.queue[current_index + 1]
        elif self.repeat_start_index is not None:
            return self.queue[self.repeat_start_index]
        else:
            return None  # end of queue reached with no repeat

    def reset_queue(self):
        self.queue = []
        self.repeat_start_index = None
        self.repeat_end_index = None
