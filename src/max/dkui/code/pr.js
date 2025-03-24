// init graphics
mgraphics.init();
mgraphics.relative_coords = 0;
mgraphics.autofill = 0;

// constants
var DEBUG = false;
var TIMESTEP = 10;
var ROLL_LEN_MS = 10000;
var MAX_NOTE_DUR_MS = 5000; // auto trim any notes longer than this
// graphical constants
var MIN_NOTE_HEIGHT = 6;
var MAX_NOTE_HEIGHT = 20;
var MIN_KEY_WIDTH = 30;
var MAX_KEY_WIDTH = 80;
var KEY_COLORS = {
  white: [0.9, 0.9, 0.9, 1.0],
  black: [0.2, 0.2, 0.2, 1.0],
  playing: [1.0, 0.5, 0.5, 1.0],
  grid: [0.7, 0.7, 0.7, 0.5],
  background: [0.15, 0.15, 0.15, 1.0],
};
var MIN_NOTE = 21; // A0
var MAX_NOTE = 108; // C8
var NOTE_RANGE = MAX_NOTE - MIN_NOTE + 1;

// variables
var windowHeight;
var windowWidth;
var keyWidth;
var noteHeight;
var whiteKeyWidth;
var blackKeyWidth;
var currentTime = 0;
var notes = [];
var activeNotes = {};
var playingNotes = Array(NOTE_RANGE).fill(0);

// note object
function Note(pitch, velocity, startTime, endTime) {
  this.pitch = pitch;
  this.velocity = velocity;
  this.startTime = startTime;
  this.endTime = endTime || null;
  this.isActive = true;
  this.isPlaying = false;
}

/**
 * Calculate UI dimensions based on window size.
 */
function calculateDimensions() {
  noteHeight = Math.min(
    Math.max(windowWidth / NOTE_RANGE, MIN_NOTE_HEIGHT),
    MAX_NOTE_HEIGHT
  );

  keyWidth = Math.min(
    Math.max(windowHeight * 0.15, MIN_KEY_WIDTH),
    MAX_KEY_WIDTH
  );

  whiteKeyWidth = keyWidth; // * 0.85;
  blackKeyWidth = keyWidth * 0.85; // * 0.6;
}

function isBlackKey(note) {
  var n = note % 12;
  return n === 1 || n === 3 || n === 6 || n === 8 || n === 10;
}

function getNoteY(note) {
  if (note < MIN_NOTE) note = MIN_NOTE;
  if (note > MAX_NOTE) note = MAX_NOTE;

  // position is calculated from bottom of screen
  return windowWidth - (note - MIN_NOTE) * noteHeight - noteHeight;
}

function timeToX(timeMS) {
  return (
    keyWidth +
    ((timeMS - (currentTime - ROLL_LEN_MS)) / ROLL_LEN_MS) *
      (windowHeight - keyWidth)
  );
}

function formatTime(ms) {
  var seconds = Math.floor(ms / 1000);
  var minutes = Math.floor(seconds / 60);
  seconds = seconds % 60;

  return minutes + ":" + (seconds < 10 ? "0" : "") + seconds;
}

function drawKeyboard() {
  // background
  mgraphics.set_source_rgba(0.1, 0.1, 0.1, 1.0);
  mgraphics.rectangle(0, 0, keyWidth, windowWidth);
  mgraphics.fill();

  // white keys
  for (var i = MIN_NOTE; i <= MAX_NOTE; i++) {
    if (!isBlackKey(i)) {
      var isPlaying = playingNotes[i - MIN_NOTE] === 1;
      mgraphics.set_source_rgba.apply(
        mgraphics,
        isPlaying ? KEY_COLORS.playing : KEY_COLORS.white
      );
      mgraphics.rectangle(0, getNoteY(i), whiteKeyWidth, noteHeight);
      mgraphics.fill();

      // border
      mgraphics.set_source_rgba(0.5, 0.5, 0.5, 1.0);
      mgraphics.rectangle(0, getNoteY(i), whiteKeyWidth, noteHeight);
      mgraphics.set_line_width(0.5);
      mgraphics.stroke();
    }
  }

  // black keys
  for (var i = MIN_NOTE; i <= MAX_NOTE; i++) {
    if (isBlackKey(i)) {
      var isPlaying = playingNotes[i - MIN_NOTE] === 1;
      mgraphics.set_source_rgba.apply(
        mgraphics,
        isPlaying ? KEY_COLORS.playing : KEY_COLORS.black
      );
      mgraphics.rectangle(0, getNoteY(i), blackKeyWidth, noteHeight);
      mgraphics.fill();
    }
  }
}

function drawGrid() {
  // draw main background
  mgraphics.set_source_rgba.apply(mgraphics, KEY_COLORS.background);
  mgraphics.rectangle(keyWidth, 0, windowHeight - keyWidth, windowWidth);
  mgraphics.fill();

  // draw horizontal grid lines for each octave
  mgraphics.set_source_rgba.apply(mgraphics, KEY_COLORS.grid);
  for (var octave = 0; octave < 9; octave++) {
    var noteNumber = 12 * octave + 12; // C notes
    if (noteNumber >= MIN_NOTE && noteNumber <= MAX_NOTE) {
      mgraphics.move_to(keyWidth, getNoteY(noteNumber));
      mgraphics.line_to(windowHeight, getNoteY(noteNumber));
      mgraphics.set_line_width(1);
      mgraphics.stroke();
    }
  }

  // draw additional grid lines for F notes
  mgraphics.set_source_rgba(0.4, 0.4, 0.4, 0.3);
  for (var octave = 0; octave < 9; octave++) {
    var noteNumber = 12 * octave + 5; // F notes
    if (noteNumber >= MIN_NOTE && noteNumber <= MAX_NOTE) {
      mgraphics.move_to(keyWidth, getNoteY(noteNumber));
      mgraphics.line_to(windowHeight, getNoteY(noteNumber));
      mgraphics.set_line_width(0.5);
      mgraphics.stroke();
    }
  }

  // draw vertical time markers every second
  for (var t = 0; t <= ROLL_LEN_MS; t += 1000) {
    var x = timeToX(currentTime - ROLL_LEN_MS + t);

    // only draw if in visible area
    if (x >= keyWidth && x <= windowHeight) {
      // draw time marker line
      mgraphics.set_source_rgba(0.4, 0.4, 0.4, 0.5);
      mgraphics.move_to(x, 0);
      mgraphics.line_to(x, windowWidth);
      mgraphics.set_line_width(1);
      mgraphics.stroke();

      // draw time labels
      if (t % 2000 === 0) {
        mgraphics.set_source_rgba(0.7, 0.7, 0.7, 1.0);
        mgraphics.select_font_face("Arial");
        mgraphics.set_font_size(10);
        if (t < ROLL_LEN_MS) {
          mgraphics.move_to(x - 5, 10);
        } else {
          mgraphics.move_to(x - 15, 10);
        }

        var relativeTimeSeconds = Math.round(-t / 1000);
        var timeLabel =
          relativeTimeSeconds === 0 ? "0" : relativeTimeSeconds.toString();
        mgraphics.text_path(timeLabel);
        mgraphics.fill();
      }
    }
  }

  // Current time marker
  var currentX = timeToX(currentTime);
  mgraphics.set_source_rgba(1.0, 0.3, 0.3, 0.8);
  mgraphics.move_to(currentX, 0);
  mgraphics.line_to(currentX, windowWidth);
  mgraphics.set_line_width(2);
  mgraphics.stroke();
}

function drawNotes() {
  var visibleStartTime = currentTime - ROLL_LEN_MS;

  for (var i = 0; i < notes.length; i++) {
    var note = notes[i];

    // skip notes that are completely before the visible time window
    if (note.endTime && note.endTime < visibleStartTime) {
      continue;
    }

    // calculate start and end positions
    var startX = Math.max(timeToX(note.startTime), keyWidth);
    var endX = note.endTime ? timeToX(note.endTime) : timeToX(currentTime);

    // skip notes that are completely after the visible time window
    if (startX > windowHeight) {
      continue;
    }

    // adjust endX if it's off-screen
    endX = Math.min(endX, windowHeight);

    // calculate note box dimensions
    var y = getNoteY(note.pitch);
    var noteWidth = Math.max(endX - startX, 2); // ensure a minimum width

    // note background is scaled by velocity
    var alpha = 0.5 + (note.velocity / 127) * 0.5;
    if (note.isActive) {
      mgraphics.set_source_rgba(0.3, 0.3, 0.3, alpha);
    } else if (note.isPlaying) {
      mgraphics.set_source_rgba(1.0, 0.5, 0.5, alpha);
    } else {
      mgraphics.set_source_rgba(0.5, 0.8, 1.0, alpha);
    }

    mgraphics.rectangle(startX, y, noteWidth, noteHeight);
    mgraphics.fill();

    // note border
    mgraphics.set_source_rgba(0.2, 0.2, 0.2, 0.8);
    mgraphics.rectangle(startX, y, noteWidth, noteHeight);
    mgraphics.set_line_width(0.5);
    mgraphics.stroke();
  }
}

function paint() {
  windowHeight = this.box.rect[2] - this.box.rect[0];
  windowWidth = this.box.rect[3] - this.box.rect[1];
  calculateDimensions(); // need to recalc on resize

  // clear any existing drawings
  mgraphics.set_source_rgba(0, 0, 0, 1);
  mgraphics.rectangle(0, 0, windowHeight, windowWidth);
  mgraphics.fill();

  // draw components
  drawGrid();
  drawNotes();
  drawKeyboard();
}

function isNoteAtKeyboard(note) {
  // calculate note's current x position
  var startX = timeToX(note.startTime);
  var endX = note.endTime ? timeToX(note.endTime) : timeToX(currentTime);

  // note is at keyboard when:
  // 1. its start position is at or before the keyboard (has reached the keyboard)
  // 2. its end position hasn't passed the keyboard yet (is still on or after the keyboard edge)
  return startX <= keyWidth && endX >= keyWidth;
}

function updatePlayingNotes() {
  // reset all playing notes
  playingNotes = Array(NOTE_RANGE).fill(0);

  // update playing status for all notes
  for (var i = 0; i < notes.length; i++) {
    var note = notes[i];
    if (isNoteAtKeyboard(note)) {
      note.isPlaying = true;

      // note index in the playingNotes array
      var noteIdx = note.pitch - MIN_NOTE;
      if (noteIdx >= 0 && noteIdx < NOTE_RANGE) {
        playingNotes[noteIdx] = 1;
      }
    } else {
      note.isPlaying = false;
    }
  }
}

function updateTime() {
  currentTime += TIMESTEP;

  // check for stuck notes and end them if they've been active too long
  for (var noteNum in activeNotes) {
    var note = activeNotes[noteNum];
    if (currentTime - note.startTime > MAX_NOTE_DUR_MS) {
      noteOff(parseInt(noteNum));
    }
  }

  // cleanup old notes that are too far in the past
  var cutoffTime = currentTime - ROLL_LEN_MS * 2;
  notes = notes.filter(function (note) {
    return note.isActive || note.endTime > cutoffTime;
  });

  // update playing notes
  updatePlayingNotes();

  mgraphics.redraw();
}

function cleanup() {
  var activeNoteNums = Object.keys(activeNotes);
  if (DEBUG) post("Cleaning up", activeNoteNums.length, "active notes\n");

  for (var i = 0; i < activeNoteNums.length; i++) {
    noteOff(parseInt(activeNoteNums[i]));
  }
}

function debug(onOff) {
  DEBUG = onOff ? true : false;
  post("Debug mode:", DEBUG ? "ON" : "OFF", "\n");
}

function noteOn(note, velocity) {
  // check if the note is already active and end it first
  if (activeNotes[note]) {
    if (DEBUG)
      post("Warning: Note", note, "already active, ending previous note\n");
    noteOff(note);
  }

  var newNote = new Note(note, velocity, currentTime);
  notes.push(newNote);
  activeNotes[note] = newNote;
  if (DEBUG)
    post("Note ON:", note, "velocity:", velocity, "time:", currentTime, "\n");
  mgraphics.redraw();
}

function noteOff(note) {
  if (activeNotes[note]) {
    activeNotes[note].endTime = currentTime;
    activeNotes[note].isActive = false;
    if (DEBUG)
      post(
        "Note OFF:",
        note,
        "duration:",
        currentTime - activeNotes[note].startTime,
        "ms\n"
      );
    delete activeNotes[note];
    mgraphics.redraw();
  } else if (DEBUG) {
    post("Warning: Received noteOff for inactive note:", note, "\n");
  }
}

function list() {
  var args = arrayfromargs(arguments);
  if (args.length >= 2) {
    var note = args[0];
    var velocity = args[1];

    if (velocity > 0) {
      noteOn(note, velocity);
    } else {
      noteOff(note);
    }
  }
}

function init() {
  this.box.size(800, 600);
  windowHeight = this.box.rect[2] - this.box.rect[0];
  windowWidth = this.box.rect[3] - this.box.rect[1];
  calculateDimensions();

  t = new Task(updateTime, this);
  t.interval = TIMESTEP;
  t.repeat();
}

init();
mgraphics.redraw();
