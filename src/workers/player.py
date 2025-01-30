import mido
import time
from queue import PriorityQueue
from datetime import datetime, timedelta

from utils import console
from .worker import Worker
from .scheduler import N_TICKS_PER_BEAT


class Player(Worker):
    td_last_note: datetime
    first_note = False
    n_notes = 0
    n_late_notes = 0

    def __init__(self, params, bpm: int, t_start: datetime):
        super().__init__(params, bpm=bpm)
        self.midi_port = mido.open_output(params.midi_port)  # type: ignore
        self.td_start = t_start
        self.td_last_note = t_start

        # if self.verbose:
        console.log(f"{self.tag} settings:\n{self.__dict__}")
        console.log(
            f"{self.tag} initialization complete, start time is {self.td_start.strftime('%H:%M:%S.%f')[:-3]}"
        )

    def play(self, queue: PriorityQueue):
        console.log(
            f"{self.tag} start time is {self.td_start.strftime('%H:%M:%S.%f')[:-3]}"
        )
        while queue.qsize() > 0:
            tt_abs, msg = queue.get()
            ts_abs = mido.tick2second(tt_abs, N_TICKS_PER_BEAT, self.tempo)
            if self.verbose:
                console.log(
                    f"{self.tag} absolute time is {tt_abs} ticks (delta is {ts_abs:.03f} seconds)"
                )

            # if self.verbose:
            #     ts_abs_message_time = mido.tick2second(
            #         tt_abs, N_TICKS_PER_BEAT, self.tempo
            #     )
            #     console.log(
            #         f"{self.tag} \ttotal time should be {self.td_start.strftime('%H:%M:%S.%f')} + {ts_abs_message_time:.02f} = {(self.td_start + timedelta(seconds=ts_abs_message_time)).strftime(('%H:%M:%S.%f'))}"
            #     )

            td_now = datetime.now()
            if not self.first_note:
                self.td_last_note = td_now
            dt_sleep = self.td_start + timedelta(seconds=ts_abs) - td_now
            if dt_sleep.total_seconds() < -0.001:
                self.n_late_notes += 1
            if self.verbose:
                dt_tag = "yellow bold" if dt_sleep.total_seconds() < -0.001 else "blue"
                console.log(
                    f"{self.tag} \tit is {td_now.strftime('%H:%M:%S.%f')} and the last note was played at {self.td_last_note.strftime('%H:%M:%S.%f')}. I will sleep for [{dt_tag}]{dt_sleep.total_seconds()}[/{dt_tag}]s"
                )
            self.td_last_note = self.td_last_note + timedelta(seconds=ts_abs)

            if dt_sleep.total_seconds() > 0:
                if self.verbose:
                    console.log(
                        f"{self.tag} \twaiting until {(td_now + dt_sleep).strftime("%H:%M:%S.%f")[:-3]} to play message: ({msg})"
                    )
                time.sleep(dt_sleep.total_seconds())

            if msg.velocity > 0 and self.verbose:
                console.log(
                    f"{self.tag} playing ({msg})\t{queue.qsize():03d} events queued"
                )
            self.midi_port.send(msg)
            self.n_notes += 1
            queue.task_done()

        # kill any remaining active notes
        for note in range(128):
            msg = mido.Message("note_off", note=note, velocity=0, channel=0)
            self.midi_port.send(msg)

        self.midi_port.close()
        console.log(f"{self.tag}[green] playback finished")
        console.log(
            f"{self.tag} [yellow bold]{self.n_late_notes}[/yellow bold]/{self.n_notes} notes were late (sent > 0.001 s after scheduled)"
        )
