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
