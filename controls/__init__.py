from pynput import keyboard
from threading import Event
from queue import Queue

from utils import console


p = "[cyan1]keybrd[/cyan1]:"

class KeyboardController:
    p = "[cyan1]keybrd[/cyan1]:"

    def __init__(self, queue: Queue, killEvent: Event):
        # self.params = params
        self.queue = queue
        self.killed = killEvent

        self.controls = {
            "b": self.rewind,
            "m": self.volume,
            ",": self.volume,
            ".": self.volume,
            "left": self.random,
            "right": self.random,
        }

        console.log(f"{self.p} keyboard controller initialized")

    def listen(self):
        console.log(f"{self.p} listening for keypresses")
        while not self.killed.is_set():
            with keyboard.Listener(
                on_press=on_press, on_release=on_release
            ) as listener:
                listener.join()

        console.log(f"{self.p} shut down")

    def volume(self, key):
        console.log(f"{self.p} [green_yellow][VOLUME][/green_yellow] key pressed {key}")
        pass

    def rewind(self, key):
        console.log(
            f"{self.p} [light_sky_blue1][REWIND][/light_sky_blue1] key pressed {key}"
        )
        pass

    def random(self, key):
        console.log(f"{self.p} [159][RANDOM] [/159] key pressed {key}")
        pass

def on_press(key):
    console.log(f"{p} key pressed: {key}")
    # for key, callback in self.controls.items():
    #     if keyboard.is_pressed(key):
    #         callback(key)

def on_release(key):
    # console.log(f"{p} key released: {key}")
    pass
