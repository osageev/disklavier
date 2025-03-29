import os
import tkinter as tk
from tkinter import ttk, filedialog
import yaml
from omegaconf import OmegaConf


class ParameterGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Disklavier Parameter Editor")
        self.root.geometry("800x600")

        self.param_options = {
            "initialization": ["kickstart", "random", "recording"],
            "seeker.mode": [
                "best",
                "random",
                "sequential",
                "graph",
            ],
            "seeker.metric": [
                "pitch-histogram",
                "specdiff",
                "clf-4note",
                "clf-speed",
                "clf-tpose",
            ],
        }

        self.params = {}
        self.param_widgets = {}

        self.create_widgets()

    def create_widgets(self):
        # File selection frame
        file_frame = ttk.Frame(self.root, padding="10")
        file_frame.pack(fill=tk.X)

        ttk.Label(file_frame, text="Parameter File:").pack(side=tk.LEFT, padx=(0, 10))
        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var, width=50).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(
            side=tk.LEFT
        )
        ttk.Button(file_frame, text="Load", command=self.load_params).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Parameter display frame with scrollbar
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas with scrollbar
        self.canvas = tk.Canvas(self.main_frame)
        scrollbar = ttk.Scrollbar(
            self.main_frame, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bottom buttons
        btn_frame = ttk.Frame(self.root, padding="10")
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="Save", command=self.save_params).pack(
            side=tk.RIGHT, padx=(10, 0)
        )
        ttk.Button(btn_frame, text="Reset", command=self.load_params).pack(
            side=tk.RIGHT
        )

    def browse_file(self):
        """
        Open file dialog to select a parameter file.
        """
        filetypes = [("YAML files", "*.yaml"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(
            title="Select Parameter File", initialdir="params", filetypes=filetypes
        )
        if filename:
            self.file_var.set(filename)

    def load_params(self):
        """
        Load parameters from the selected file.
        """
        filepath = self.file_var.get()
        if not filepath or not os.path.exists(filepath):
            filepath = "params/disklavier.yaml"  # Default

        try:
            self.params = OmegaConf.load(filepath)
            self.display_params()
        except Exception as e:
            print(f"Error loading parameters: {e}")

    def display_params(self):
        """
        Display parameters in the GUI.
        """
        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.param_widgets = {}

        # Create header
        header_frame = ttk.Frame(self.scrollable_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header_frame, text="Parameter", width=30).pack(side=tk.LEFT)
        ttk.Label(header_frame, text="Value", width=40).pack(side=tk.LEFT)

        # Add separator
        ttk.Separator(self.scrollable_frame, orient="horizontal").pack(
            fill=tk.X, pady=5
        )

        if len(self.params.keys()) == 0:
            # Show message if no parameters are loaded
            msg_frame = ttk.Frame(self.scrollable_frame)
            msg_frame.pack(fill=tk.X, pady=10)
            ttk.Label(
                msg_frame, text="No parameters loaded. Please load a parameter file."
            ).pack(anchor=tk.CENTER)
            return

        # Display top-level parameters
        for key in self.params.keys():
            if isinstance(self.params[key], (int, float, str, bool)):
                self.add_param_row(key, self.params[key])
            elif isinstance(self.params[key], (dict, OmegaConf)):
                # Create section header
                section_frame = ttk.Frame(self.scrollable_frame)
                section_frame.pack(fill=tk.X, pady=(10, 5))
                ttk.Label(
                    section_frame, text=key, font=("TkDefaultFont", 10, "bold")
                ).pack(anchor=tk.W)

                # Add nested parameters
                for subkey in self.params[key].keys():
                    full_key = f"{key}.{subkey}"
                    self.add_param_row(full_key, self.params[key][subkey])

    def add_param_row(self, key, value):
        """
        Add a parameter row to the GUI.

        Parameters
        ----------
        key : str
            Parameter key.
        value : any
            Parameter value.
        """
        frame = ttk.Frame(self.scrollable_frame)
        frame.pack(fill=tk.X, pady=2)

        ttk.Label(frame, text=key, width=30).pack(side=tk.LEFT)

        # Use appropriate widget based on parameter type and options
        if key in self.param_options:
            var = tk.StringVar(value=str(value))
            widget = ttk.Combobox(
                frame, textvariable=var, values=self.param_options[key], width=38
            )
            widget.pack(side=tk.LEFT)
            self.param_widgets[key] = var
        elif isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            widget = ttk.Checkbutton(frame, variable=var)
            widget.pack(side=tk.LEFT)
            self.param_widgets[key] = var
        elif isinstance(value, (int, float)):
            var = tk.StringVar(value=str(value))
            widget = ttk.Entry(frame, textvariable=var, width=40)
            widget.pack(side=tk.LEFT)
            self.param_widgets[key] = var
        else:
            var = tk.StringVar(value=str(value))
            widget = ttk.Entry(frame, textvariable=var, width=40)
            widget.pack(side=tk.LEFT)
            self.param_widgets[key] = var

    def save_params(self):
        """
        Save parameters to the selected file.
        """
        if len(self.params.keys()) == 0:
            return

        # Update params from widgets
        for key, var in self.param_widgets.items():
            value = var.get()

            # Convert types
            if isinstance(self.get_param_value(key), int):
                try:
                    value = int(value)
                except ValueError:
                    continue
            elif isinstance(self.get_param_value(key), float):
                try:
                    value = float(value)
                except ValueError:
                    continue
            elif isinstance(self.get_param_value(key), bool):
                value = bool(value)

            # Set value in params
            self.set_param_value(key, value)

        # Save to file
        filepath = self.file_var.get()
        if not filepath:
            filepath = filedialog.asksaveasfilename(
                title="Save Parameter File",
                initialdir="params",
                filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            )
            if not filepath:
                return

        # Convert to regular dict and save
        with open(filepath, "w") as f:
            yaml.dump(OmegaConf.to_container(self.params), f, default_flow_style=False)

    def get_param_value(self, key):
        """
        Get parameter value from the config.

        Parameters
        ----------
        key : str
            Parameter key.

        Returns
        -------
        any
            Parameter value.
        """
        if len(self.params.keys()) == 0:
            return None

        if "." in key:
            section, subkey = key.split(".", 1)
            return self.params[section][subkey]
        return self.params[key]

    def set_param_value(self, key, value):
        """
        Set parameter value in the config.

        Parameters
        ----------
        key : str
            Parameter key.
        value : any
            Parameter value.
        """
        if len(self.params.keys()) == 0:
            return

        if "." in key:
            section, subkey = key.split(".", 1)
            self.params[section][subkey] = value
        else:
            self.params[key] = value


def main():
    """
    Run the parameter GUI.
    """
    root = tk.Tk()
    app = ParameterGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
