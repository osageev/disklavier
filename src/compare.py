import os
import mido
import numpy as np
import pretty_midi
from rich import print
from utils import basename
from itertools import product
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from queue import PriorityQueue
from argparse import ArgumentParser
from datetime import datetime, timedelta

from utils.midi import trim_piano_roll, upsample_piano_roll
from utils.novelty import gen_ssm_and_novelty
from workers import Seeker, Scheduler

CONSTANTS = {
    "system": "test",
    "bpm": 60,
    "ticks_per_beat": 220,
    "n_beats_per_segment": 8,
    "n_transitions": 16,
    "n_min_queue_length": 100,
    "startup_delay": 10,
    "initialization": "kickstart",
    "kickstart_path": "data/datasets/test/test/intervals-060-09_1_t00s00.mid",
}

OPTIONS = {
    "dataset": ["20250110", "20250320", "20250320-c10", "20250320-c100"],
    "mode": ["best", "graph"],
    "metric": ["pitch-histogram", "specdiff", "clf-4note", "clf-speed", "clf-tpose", "clamp"],
    "graph_steps": [5, 9, 17],
}


def main(args):
    start_time = datetime.now()
    # get all possible combinations of settings
    keys = list(OPTIONS.keys())
    values = list(OPTIONS.values())
    product_combo = list(product(*values))
    settings = [dict(zip(keys, combo)) for combo in product_combo]
    print(f"running with {len(settings)} settings combinations")

    clean_conf = OmegaConf.load("params/live_template.yaml")
    for i, setting in enumerate(settings):
        params = OmegaConf.merge(clean_conf, CONSTANTS, setting)
        params.tables = os.path.join(args.data_dir, params.dataset)
        params.dataset_path = os.path.join(params.tables, "augmented")
        params.scheduler.n_beats_per_segment = params.n_beats_per_segment
        params.seeker.system = params.system
        params.scheduler.system = params.system
        print(f"running with settings:")
        print(OmegaConf.to_yaml(params))

        start_time = datetime.now()
        start_str = start_time.strftime("%Y%m%d_%H%M%S")
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
        pf_next_file = params.kickstart_path
        seeker.played_files.append(pf_next_file)
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
            ts_seg_len, ts_seg_start = scheduler.enqueue_midi(pf_next_file, q_playback)
            pf_next_file, similarity = seeker.get_next()
            similarities.append(similarity)
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
        num_notes = 0
        note_track.name = f"{i}"
        while not q_playback.empty():
            tt_msg, msg = q_playback.get()
            if num_notes == 0:
                msg.time = 0
            else:
                msg.time = tt_msg - tt_tracker
            tt_tracker = tt_msg
            note_track.append(msg)
            num_notes += 1
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
        for i in range(len(scheduler.queued_files)):
            transition_track.append(
                mido.MetaMessage(
                    "text",
                    text=f"{basename(scheduler.queued_files[i])}",
                    time=mido.second2tick(
                        scheduler.ts_transitions[i],
                        midi.ticks_per_beat,
                        mido.bpm2tempo(params.bpm),
                    ),
                )
            )
        midi.tracks.append(transition_track)
        midi.save(midi_file)

        # save PR
        sr = 100
        piano_roll = pretty_midi.PrettyMIDI(midi_file).get_piano_roll(sr)
        piano_roll_path = os.path.join(output_dir, f"{start_str}-{i}-pr.png")

        # create figure with minimal padding
        plt.figure(figsize=(12, 4))

        # trim and upsample piano roll to fixed resolution
        trimmed_roll = trim_piano_roll(piano_roll)
        upsampled_roll = upsample_piano_roll(trimmed_roll)
        plt.imshow(upsampled_roll, aspect="auto", origin="lower", cmap="gray_r")

        # add vertical lines at transition times
        for transition in transition_track:
            # convert ticks to seconds
            time_in_seconds = mido.tick2second(
                transition.time, midi.ticks_per_beat, mido.bpm2tempo(params.bpm)
            )
            # convert seconds to piano roll samples and scale to upsampled width
            x_pos = int(time_in_seconds * sr * (1200 / piano_roll.shape[1]))
            plt.axvline(x=x_pos, color="blue", alpha=0.5, linewidth=0.5)

        # generate novelty curve
        ssm, novelty = gen_ssm_and_novelty(midi_file)
        # account for downsampling factor of 10 in novelty curve
        scaling_factor = upsampled_roll.shape[1] / piano_roll.shape[1]
        x = (
            np.arange(len(novelty)) * scaling_factor * 10
        )  # stretch x-axis to match piano roll width
        plt.plot(
            x,
            (1 - novelty / novelty.max()) * 17,
            "g",
            linewidth=1.0,
            alpha=0.7,
        )

        plt.xticks([])
        plt.yticks([])
        plt.axis("off")

        # remove all padding and make the plot fill the figure
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        plt.savefig(piano_roll_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        # save SSM
        plt.figure(figsize=(16, 16))
        plt.imshow(ssm / ssm.max(), cmap="magma")
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        plt.savefig(os.path.join(output_dir, f"{start_str}-{i}-ssm.png"))
        plt.close()

    print(f"comparison completed in {datetime.now() - start_time}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", type=str, default="/media/scratch/sageev-midi"
    )
    args = parser.parse_args()

    main(args)
