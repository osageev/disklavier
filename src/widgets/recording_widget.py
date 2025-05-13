import PySide6.QtGui as QtGui
import PySide6.QtCore as QtCore
import PySide6.QtWidgets as QtWidgets


class RecordingWidget(QtWidgets.QWidget):
    """
    widget to display recording progress and augmentation status.

    parameters
    ----------
    parent : QtWidgets.QWidget
        parent widget, typically mainwindow.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """
        initialize the user interface elements.
        """
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        main_layout.setSpacing(20)

        # --- font --- #
        large_font = QtGui.QFont()
        large_font.setPointSize(24)
        medium_font = QtGui.QFont()
        medium_font.setPointSize(18)

        # --- recording time --- #
        time_label = QtWidgets.QLabel("Time Elapsed:")
        time_label.setFont(medium_font)
        time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.seconds_label = QtWidgets.QLabel("0.00 s")
        self.seconds_label.setFont(large_font)
        self.seconds_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        beats_label = QtWidgets.QLabel("Beats Elapsed:")
        beats_label.setFont(medium_font)
        beats_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.beats_label = QtWidgets.QLabel("0.0")
        self.beats_label.setFont(large_font)
        self.beats_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # --- augmentation progress --- #
        aug_label = QtWidgets.QLabel("Augmentation Progress:")
        aug_label.setFont(medium_font)
        aug_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)  # Default max, will be updated
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setFixedHeight(30)
        self.progress_bar.setFixedWidth(400)

        # --- layout assembly --- #
        main_layout.addStretch()
        main_layout.addWidget(time_label)
        main_layout.addWidget(self.seconds_label)
        main_layout.addSpacing(10)
        main_layout.addWidget(beats_label)
        main_layout.addWidget(self.beats_label)
        main_layout.addSpacing(30)
        main_layout.addWidget(aug_label)
        main_layout.addWidget(self.progress_bar, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        main_layout.addStretch()

    @QtCore.Slot(float, float)
    def update_recording_time(self, seconds: float, beats: float):
        """
        update the displayed recording time and beats.

        parameters
        ----------
        seconds : float
            elapsed seconds.
        beats : float
            elapsed beats.
        """
        self.seconds_label.setText(f"{seconds:.2f} s")
        self.beats_label.setText(f"{beats:.1f}")

    @QtCore.Slot(int)
    def init_augmentation_progress(self, total_steps: int):
        """
        initialize the augmentation progress bar.

        parameters
        ----------
        total_steps : int
            total number of steps (embeddings to calculate).
        """
        self.progress_bar.setMaximum(total_steps)
        self.progress_bar.setValue(0)

    @QtCore.Slot()
    def update_augmentation_progress(self):
        """
        increment the augmentation progress bar.
        """
        current_value = self.progress_bar.value()
        self.progress_bar.setValue(current_value + 1)

    def reset_widget(self):
        """
        reset the widget to its initial state.
        """
        self.seconds_label.setText("0.00 s")
        self.beats_label.setText("0.0")
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)  # Reset to default or 0
