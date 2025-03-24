mgraphics.init();
mgraphics.relative_coords = 0;
mgraphics.autofill = 0;

var windowHeight;
var windowWidth;
var keyWidth; // Now calculated based on window width
var noteHeight; // Now calculated based on window height
var whiteKeyWidth; // Now calculated based on keyWidth
var blackKeyWidth; // Now calculated based on keyWidth

var ROLL_LEN_MS = 10000; // 10 seconds in milliseconds
var scrollSpeed = 20; // pixels per second
var currentTime = 0;
var notes = [];
var activeNotes = {};

// Min/max sizes to ensure legibility
var MIN_NOTE_HEIGHT = 6;
var MAX_NOTE_HEIGHT = 20;
var MIN_KEY_WIDTH = 40;
var MAX_KEY_WIDTH = 80;

var KEY_COLORS = {
  white: [0.9, 0.9, 0.9, 1.0],
  black: [0.2, 0.2, 0.2, 1.0],
  active: [1.0, 0.5, 0.5, 1.0],
  grid: [0.7, 0.7, 0.7, 0.5],
  background: [0.15, 0.15, 0.15, 1.0],
};

var g = new Global("piano_roll");

// Standard piano range (88 keys)
var MIN_NOTE = 21; // A0
var highestNote = 108; // C8
var noteRange = highestNote - MIN_NOTE + 1;

/**
 * Note object constructor with MIDI note properties.
 *
 * Parameters
 * ----------
 * pitch : int
 *     MIDI note number (0-127).
 * velocity : int
 *     MIDI velocity (0-127).
 * startTime : int
 *     Start time in milliseconds.
 * endTime : int
 *     End time in milliseconds (defaults to null for active notes).
 */
function Note(pitch, velocity, startTime, endTime) {
  this.pitch = pitch;
  this.velocity = velocity;
  this.startTime = startTime;
  this.endTime = endTime || null;
  this.isActive = true;
}

/**
 * Calculate UI dimensions based on window size.
 */
function calculateDimensions() {
  // Calculate note height based on available height and note range
  noteHeight = Math.min(
    Math.max(windowWidth / noteRange, MIN_NOTE_HEIGHT),
    MAX_NOTE_HEIGHT
  );

  // Calculate key width based on window width (10-15% of total width)
  keyWidth = Math.min(
    Math.max(windowHeight * 0.12, MIN_KEY_WIDTH),
    MAX_KEY_WIDTH
  );

  // Calculate white and black key widths based on key width
  whiteKeyWidth = keyWidth * 0.85;
  blackKeyWidth = keyWidth * 0.6;
}

function isBlackKey(note) {
  var n = note % 12;
  return n === 1 || n === 3 || n === 6 || n === 8 || n === 10;
}

function getNoteY(note) {
  // Map note to position within the piano range
  if (note < MIN_NOTE) note = MIN_NOTE;
  if (note > highestNote) note = highestNote;

  // Calculate position from bottom of screen
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
  // Draw the keyboard for standard piano range (notes 21-108)

  // Background for keyboard
  mgraphics.set_source_rgba(0.1, 0.1, 0.1, 1.0);
  mgraphics.rectangle(0, 0, keyWidth, windowWidth);
  mgraphics.fill();

  // Draw white keys first
  for (var i = MIN_NOTE; i <= highestNote; i++) {
    if (!isBlackKey(i)) {
      var isActive = activeNotes[i] !== undefined;
      mgraphics.set_source_rgba.apply(
        mgraphics,
        isActive ? KEY_COLORS.active : KEY_COLORS.white
      );
      mgraphics.rectangle(0, getNoteY(i), whiteKeyWidth, noteHeight);
      mgraphics.fill();

      // Add borders to white keys
      mgraphics.set_source_rgba(0.5, 0.5, 0.5, 1.0);
      mgraphics.rectangle(0, getNoteY(i), whiteKeyWidth, noteHeight);
      mgraphics.set_line_width(0.5);
      mgraphics.stroke();
    }
  }

  // Draw black keys on top
  for (var i = MIN_NOTE; i <= highestNote; i++) {
    if (isBlackKey(i)) {
      var isActive = activeNotes[i] !== undefined;
      mgraphics.set_source_rgba.apply(
        mgraphics,
        isActive ? KEY_COLORS.active : KEY_COLORS.black
      );
      mgraphics.rectangle(0, getNoteY(i), blackKeyWidth, noteHeight);
      mgraphics.fill();
    }
  }

  // Draw octave labels (C notes)
  // mgraphics.set_source_rgba(1, 1, 1, 1);
  // mgraphics.select_font_face("Arial");
  // mgraphics.set_font_size(8);

  // for (var octave = 0; octave < 9; octave++) {
  //   var noteNumber = 12 * octave + 12; // C notes (C0 = 12, C1 = 24, etc.)
  //   if (noteNumber >= lowestNote && noteNumber <= highestNote) {
  //     mgraphics.move_to(keyWidth - 15, getNoteY(noteNumber) + 7);
  //     mgraphics.text_path("C" + octave);
  //     mgraphics.fill();
  //   }
  // }
}

function drawGrid() {
  // Draw main background
  mgraphics.set_source_rgba.apply(mgraphics, KEY_COLORS.background);
  mgraphics.rectangle(keyWidth, 0, windowHeight - keyWidth, windowWidth);
  mgraphics.fill();

  // Draw horizontal grid lines for each octave (C notes)
  mgraphics.set_source_rgba.apply(mgraphics, KEY_COLORS.grid);
  for (var octave = 0; octave < 9; octave++) {
    var noteNumber = 12 * octave + 12; // C notes
    if (noteNumber >= MIN_NOTE && noteNumber <= highestNote) {
      mgraphics.move_to(keyWidth, getNoteY(noteNumber));
      mgraphics.line_to(windowHeight, getNoteY(noteNumber));
      mgraphics.set_line_width(1);
      mgraphics.stroke();
    }
  }

  // Draw additional grid lines for F notes (lighter)
  mgraphics.set_source_rgba(0.4, 0.4, 0.4, 0.3);
  for (var octave = 0; octave < 9; octave++) {
    var noteNumber = 12 * octave + 5; // F notes
    if (noteNumber >= MIN_NOTE && noteNumber <= highestNote) {
      mgraphics.move_to(keyWidth, getNoteY(noteNumber));
      mgraphics.line_to(windowHeight, getNoteY(noteNumber));
      mgraphics.set_line_width(0.5);
      mgraphics.stroke();
    }
  }

  // Draw vertical time markers (every second)
  for (var t = 0; t <= ROLL_LEN_MS; t += 1000) {
    var x = timeToX(currentTime - ROLL_LEN_MS + t);

    // Only draw if in visible area
    if (x >= keyWidth && x <= windowHeight) {
      // Draw time marker line
      mgraphics.set_source_rgba(0.4, 0.4, 0.4, 0.5);
      mgraphics.move_to(x, 0);
      mgraphics.line_to(x, windowWidth);
      mgraphics.set_line_width(1);
      mgraphics.stroke();

      // Draw time label
      // if (t % 5000 === 0) {
      //   // Every 5 seconds
      //   mgraphics.set_source_rgba(0.7, 0.7, 0.7, 1.0);
      //   mgraphics.select_font_face("Arial");
      //   mgraphics.set_font_size(9);
      //   mgraphics.move_to(x - 8, 10);
      //   mgraphics.text_path(formatTime(currentTime - displayTime + t));
      //   mgraphics.fill();
      // }
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

/**
 * Draw the note bars for all visible notes.
 */
function drawNotes() {
  var visibleStartTime = currentTime - ROLL_LEN_MS;

  for (var i = 0; i < notes.length; i++) {
    var note = notes[i];

    // Skip notes that are completely before the visible time window
    if (note.endTime && note.endTime < visibleStartTime) {
      continue;
    }

    // Calculate start and end positions
    var startX = Math.max(timeToX(note.startTime), keyWidth);
    var endX = note.endTime ? timeToX(note.endTime) : timeToX(currentTime);

    // Skip notes that are completely after the visible time window
    if (startX > windowHeight) {
      continue;
    }

    // Adjust endX if it's off-screen
    endX = Math.min(endX, windowHeight);

    // Calculate note box dimensions
    var y = getNoteY(note.pitch);
    var noteWidth = Math.max(endX - startX, 2); // Ensure a minimum width

    // Draw note background
    var alpha = 0.5 + (note.velocity / 127) * 0.5; // Scale by velocity

    // Use a color that indicates whether the note is still active
    if (note.isActive) {
      mgraphics.set_source_rgba(1.0, 0.5, 0.5, alpha);
    } else {
      mgraphics.set_source_rgba(0.5, 0.8, 1.0, alpha);
    }

    mgraphics.rectangle(startX, y, noteWidth, noteHeight);
    mgraphics.fill();

    // Draw note border
    mgraphics.set_source_rgba(0.2, 0.2, 0.2, 0.8);
    mgraphics.rectangle(startX, y, noteWidth, noteHeight);
    mgraphics.set_line_width(0.5);
    mgraphics.stroke();
  }
}

/**
 * Main paint function to draw all UI elements.
 */
function paint() {
  windowHeight = this.box.rect[2] - this.box.rect[0];
  windowWidth = this.box.rect[3] - this.box.rect[1];

  // Recalculate dimensions when the window is resized
  calculateDimensions();

  // Clear any existing drawings
  mgraphics.set_source_rgba(0, 0, 0, 1);
  mgraphics.rectangle(0, 0, windowHeight, windowWidth);
  mgraphics.fill();

  // Draw components
  drawGrid();
  drawNotes();
  drawKeyboard();

  // Draw title and time display
  // mgraphics.set_source_rgba(1, 1, 1, 1);
  // mgraphics.select_font_face("Arial Black");
  // mgraphics.set_font_size(12);
  // mgraphics.move_to(keyWidth + 10, 15);
  // mgraphics.text_path("Piano Roll - Last " + displayTime / 1000 + " seconds");
  // mgraphics.fill();
}

/**
 * Update time and scroll the view.
 */
function updateTime() {
  currentTime += 100; // Update every 100ms

  // Cleanup old notes that are too far in the past
  var cutoffTime = currentTime - ROLL_LEN_MS * 2;
  notes = notes.filter(function (note) {
    return note.isActive || note.endTime > cutoffTime;
  });

  mgraphics.redraw();
}

/**
 * Handle note on message (note, velocity).
 *
 * Parameters
 * ----------
 * note : int
 *     MIDI note number (0-127).
 * velocity : int
 *     MIDI velocity (0-127).
 */
function noteOn(note, velocity) {
  if (velocity > 0) {
    var newNote = new Note(note, velocity, currentTime);
    notes.push(newNote);
    activeNotes[note] = newNote;
  } else {
    // Note-on with velocity 0 is equivalent to note-off
    noteOff(note);
  }
  mgraphics.redraw();
}

/**
 * Handle note off message.
 *
 * Parameters
 * ----------
 * note : int
 *     MIDI note number (0-127).
 */
function noteOff(note) {
  if (activeNotes[note]) {
    activeNotes[note].endTime = currentTime;
    activeNotes[note].isActive = false;
    delete activeNotes[note];
    mgraphics.redraw();
  }
}

/**
 * Handle incoming lists (assumed to be note, velocity pairs).
 */
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

/**
 * Check if a note is currently at the piano keyboard position.
 *
 * Parameters
 * ----------
 * note : object
 *     Note object to check.
 *
 * Returns
 * -------
 * boolean
 *     True if the note is currently at the keyboard position.
 */
function isNoteAtKeyboard(note) {
  // Calculate note's current x position
  var startX = timeToX(note.startTime);
  var endX = note.endTime
    ? timeToX(note.endTime)
    : timeToX(currentTime + ROLL_LEN_MS);

  // Note is at keyboard when its left edge has reached the keyboard
  // and its right edge hasn't completely passed it yet
  return startX <= keyWidth && endX >= keyWidth;
}

/**
 * Initialize the UI.
 */
function init() {
  this.box.size(800, 600);

  // Set initial dimensions
  windowHeight = this.box.rect[2] - this.box.rect[0];
  windowWidth = this.box.rect[3] - this.box.rect[1];
  calculateDimensions();

  // Start the timer for updating time
  t = new Task(updateTime, this);
  t.interval = 100; // Update every 100ms
  t.repeat();
}

init();
mgraphics.redraw();
