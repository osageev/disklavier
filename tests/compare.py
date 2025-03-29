import os
import mido
from rich import print
import pretty_midi
from itertools import product
from omegaconf import OmegaConf
from queue import PriorityQueue
from argparse import ArgumentParser
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from src.utils import basename
from src.workers import Seeker, Scheduler

CONSTANTS = {
    "bpm": 60,
    "ticks_per_beat": 220,
    "n_beats_per_segment": 8,
    "n_transitions": 16,
    "n_min_queue_length": 100,
    "startup_delay": 10,
    "initialization": "kickstart",
    "kickstart_path": "data/datasets/test/intervals-060-09_1_t00s00.mid",
}

OPTIONS = {
    "dataset": ["20250110", "20250320", "20250320-c10", "20250320-c100"],
    "mode": ["best", "graph"],
    "metric": ["pitch-histogram", "specdiff", "clf-4note", "clf-speed", "clf-tpose"],
    "graph_steps": [5, 9],
}


def main(args):
    # get all possible combinations of settings
    keys = list(OPTIONS.keys())
    values = list(OPTIONS.values())
    product_combo = list(product(*values))
    settings = [dict(zip(keys, combo)) for combo in product_combo]
    print(f"running with {len(settings)} settings combinations")

    clean_conf = OmegaConf.load("params/live_template.yaml")
    for i, setting in enumerate(settings[:3]):
        params = OmegaConf.merge(clean_conf, CONSTANTS, setting)
        params.tables = os.path.join(args.data_dir, params.dataset)
        params.dataset_path = os.path.join(params.tables, "augmented")
        params.scheduler.n_beats_per_segment = params.n_beats_per_segment
        print(f"running with settings:")
        print(OmegaConf.to_yaml(params))

        start_time = datetime.now()
        start_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"data/tests/compare/{start_str}/{i}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"output will be saved to '{output_dir}'")
        p_log = os.path.join(
            output_dir,
            f"{start_str}",
        )
        p_playlist = os.path.join(p_log, "playlist")
        os.makedirs(p_log)
        os.makedirs(p_playlist)

        # init workers
        seeker = Seeker(
            params.seeker, params.tables, params.dataset_path, p_playlist, params.bpm
        )
        scheduler = Scheduler(
            params.scheduler,
            params.bpm,
            p_log,
            p_playlist,
            start_time,
            params.n_transitions,
            params.initialization == "recording",
        )

        # ready system
        pf_seed = params.kickstart_path
        seeker.played_files.append(pf_seed)
        q_playback = PriorityQueue()
        td_start = datetime.now() + timedelta(seconds=5)
        n_files = 1
        scheduler.td_start = td_start

        if scheduler.init_schedule(
            os.path.join(output_dir, "schedule.mid"),
        ):
            print(f"successfully initialized recording")
        else:
            print(f"[red]error initializing recording, exiting")
            raise FileExistsError("Couldn't initialize MIDI recording file")

        # load queue
        similarities = []
        while n_files < params.n_transitions:
            pf_next_file, similarity = seeker.get_next()
            similarities.append(similarity)
            ts_seg_len, ts_seg_start = scheduler.enqueue_midi(pf_next_file, q_playback)
            print(
                f"added file {n_files}/{params.n_transitions} '{pf_next_file}' to queue at {ts_seg_start:.03f} s"
            )
            n_files += 1

        # build midi file
        tt_tracker = 0
        midi_file = os.path.join(output_dir, f"{start_str}-{i}.mid")
        midi = mido.MidiFile(ticks_per_beat=params.ticks_per_beat)
        # notes
        note_track = mido.MidiTrack()
        note_track.append(
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(params.bpm))
        )
        note_track.name = i
        while not q_playback.empty():
            tt_msg, msg = q_playback.get()
            msg.time = tt_msg - tt_tracker
            tt_tracker = tt_msg
            note_track.append(msg)
        midi.tracks.append(note_track)

        # beats
        beat_track = mido.MidiTrack()
        beat_track.name = "beats"
        for i in range(params.n_beats_per_segment + 1):
            beat_track.append(
                mido.MetaMessage(
                    "text",
                    text=f"beat {i}",
                    time=mido.second2tick(
                        60 / params.bpm if i > 0 else 0,
                        midi.ticks_per_beat,
                        mido.bpm2tempo(params.bpm),
                    ),
                )
            )
        midi.tracks.append(beat_track)

        # transitions
        transition_track = mido.MidiTrack()
        transition_track.name = "transitions"
        for i, ts_transition in enumerate(scheduler.ts_transitions):
            transition_track.append(
                mido.MetaMessage(
                    "text",
                    text=f"{basename(scheduler.queued_files[i])}",
                    time=mido.second2tick(
                        ts_transition, midi.ticks_per_beat, mido.bpm2tempo(params.bpm)
                    ),
                )
            )
        midi.tracks.append(transition_track)
        midi.save(midi_file)

        # save PR
        piano_roll = pretty_midi.PrettyMIDI(midi_file).get_piano_roll()
        piano_roll_path = os.path.join(output_dir, f"{start_str}-{i}-pr.png")
        plt.figure(figsize=(12, 4))
        plt.imshow(piano_roll, aspect="auto", origin="lower", cmap="gray_r")
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(piano_roll_path)
        plt.close()
        raise Exception("stop here")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", type=str, default="/media/scratch/sageev-midi"
    )
    args = parser.parse_args()

    main(args)
