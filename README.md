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
