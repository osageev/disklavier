import mido
import time
import socket
import struct
from queue import PriorityQueue
from threading import Thread, Event
from datetime import datetime, timedelta

from utils import console
from utils.midi import TICKS_PER_BEAT
from .worker import Worker


class Max(Worker):
    """
    Sends note data to Max
    """

    first_note = False

    def __init__(self, params, bpm: int, td_start: datetime, pf_max: str):
        super().__init__(params, bpm=bpm)
        self.q_max = PriorityQueue()
        self.midi_port = mido.open_output(self.params.midi_port)  # type: ignore
        self.td_start = td_start
        self.pf_max = pf_max

        # initialize UDP socket
        self.udp_port = 7400
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.track = mido.MidiTrack()
        self.track.append(mido.MetaMessage("track_name", name="max", time=0))
        self.track.append(
            mido.MetaMessage(
                type="set_tempo",
                tempo=self.tempo,
                time=0,
            )
        )

    def play(self, queue: PriorityQueue):
        """
        Starts the thread which sends note data to Max

        Parameters
        ----------
        queue : PriorityQueue
            The queue to read the MIDI from

        Returns
        -------
        Event
            The stop event
        """
        self.q_max = queue
        self.stop_event = Event()
        self.max_thread = Thread(target=self.run, name="max", daemon=True)
        self.max_thread.start()

        return self.stop_event

    def stop(self):
        self.stop_event.set()
        if self.max_thread is not None:
            self.max_thread.join(0.1)
        self.midi_port.close()
        self.udp_socket.close()

        if self.max_thread.is_alive():
            self.max_thread.join(0.1)
            console.log(
                f"{self.tag}[yellow bold] max thread is still running[/yellow bold]"
            )

        midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
        midi.tracks.append(self.track)
        midi.save(self.pf_max)

    def send_udp(self, message):
        """
        send a message via udp as an osc packet

        parameters
        ----------
        message : str
            message to send via udp

        returns
        -------
        none
        """
        if isinstance(message, str):
            # format as osc packet to ensure multiple of 4 bytes
            address = "/midi"
            # format address with null padding to multiple of 4
            address_padded = address.encode("utf-8") + b"\0" * (4 - (len(address) % 4))
            if len(address) % 4 == 0:
                address_padded += b"\0" * 4

            # format type tags (s for string)
            type_tag = ",s"
            type_tag_padded = type_tag.encode("utf-8") + b"\0" * (
                4 - (len(type_tag) % 4)
            )
            if len(type_tag) % 4 == 0:
                type_tag_padded += b"\0" * 4

            # format string argument with null padding
            arg_encoded = message.encode("utf-8")
            padding = 4 - (len(arg_encoded) % 4)
            if padding == 4:
                padding = 0
            arg_padded = arg_encoded + b"\0" * padding

            # combine all parts
            packet = address_padded + type_tag_padded + arg_padded

            self.udp_socket.sendto(packet, ("127.0.0.1", self.udp_port))
            console.log(f"{self.tag} sent UDP OSC packet: {message}")

    def run(self):
        console.log(
            f"{self.tag} start time is {self.td_start.strftime('%H:%M:%S.%f')[:-3]}"
        )
        last_note_time = self.td_start

        while not self.stop_event.is_set():
            tt_abs, msg = self.q_max.get()
            ts_abs = mido.tick2second(tt_abs, TICKS_PER_BEAT, self.tempo)

            if self.verbose:
                console.log(
                    f"{self.tag} absolute time is {tt_abs} ticks (delta is {ts_abs:.03f} seconds)"
                )

            td_now = datetime.now()
            dt_sleep = self.td_start + timedelta(seconds=ts_abs) - td_now

            if dt_sleep.total_seconds() > 0:
                if self.verbose:
                    console.log(
                        f"{self.tag} \twaiting until {(td_now + dt_sleep).strftime("%H:%M:%S.%f")[:-3]} to play message: ({msg})"
                    )
                time.sleep(dt_sleep.total_seconds())

            console.log(f"{self.tag} sending message: {msg}")
            self.midi_port.send(msg)

            # also send a simple UDP message
            self.send_udp(
                f"{msg.type}:{msg.note if hasattr(msg, 'note') else 0}:{msg.velocity if hasattr(msg, 'velocity') else 0}"
            )

            # record midi
            current_time = datetime.now()
            new_time = mido.second2tick(
                (current_time - last_note_time).total_seconds(),
                TICKS_PER_BEAT,
                self.bpm,
            )
            # msg = msg.copy(time=new_time if new_time > 0 else 0)
            self.track.append(msg)
            last_note_time = current_time

        # turn off notes & close connection
        for note in range(128):
            msg = mido.Message("note_off", note=note, velocity=0, channel=0)
            self.midi_port.send(msg)
        self.midi_port.close()

        console.log(f"{self.tag}[green] playback finished")
