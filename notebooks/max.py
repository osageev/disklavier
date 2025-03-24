import mido
import time

with mido.open_output("to Max 1") as midi_out:
    while True:
        msg = mido.Message("control_change", control=1, value=int(time.time() % 127))
        midi_out.send(msg)
        time.sleep(1)
