# disklavier ostinato

repository for disklavier things.

## variable prefix list

These are mainly for standard types, things like, e.g. a `mido.MidiFile` will generally just be named `midi_file`

| prefix  |   meaning |
|--- |--- |
|   `t_` |   time (try to remove) |
|   `td_` | `datetime.datetime` object   |
|   `ts_` |   time in seconds |
|   `tt_` | time in ticks   |
|  `dt_`  |  delta time, probably a `datetime.timedelta`  |
|  `n_`  | some sort of countable integer, short for number   |
|   `p` |   path to a folder  |
|   `pf` | path to a file, includes filename   |
|   `q` |   queue, probably `queue.PriorityQueue` |

## TODOs

1. fix midi recording
2. use recorder to add audio recording
3. add command-line feature for sageev to be able to kickstart using previous recording
4. add metric as command-line option override
5. add searching using all (non-averaged) embeddings from panther
   1. stop averaging on panther, pass back all embeddings
   2. average on local machine
   3. search index as one big chunk
   4. use best result


## TODO List

- ./src/workers/seeker.py:40:    # TODO: i forget what this does tbh, rewrite entire playlist mode
- ./src/workers/seeker.py:107:            exit()  # TODO: handle this better (return an error, let main handle it)
- ./src/workers/seeker.py:222:        # TODO: move embedding normalization to dataset generation
- ./src/workers/seeker.py:299:              # TODO: modify this to get nearest neighbor from different track
- ./src/workers/seeker.py:491:        # TODO: modify this to work from current playlist paradigm
- ./src/workers/player.py:91:        # TODO: move this to class variables
- ./src/build_dataset.py:116:    max_up = 108 - highest_note  # TODO double-check this IRL
- ./src/main.py:135:    # TODO: fix ~1-beat delay in audio recording startup
- ./src/main.py:191:        # TODO: move this to be managed by scheduler and track scheduler state instead
- ./src/main.py:194:                # TODO: get first match sooner if using a recording
- ./src/main.py:399:        exit()  # TODO: handle this better
