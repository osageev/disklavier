mgraphics.init();
mgraphics.relative_coords = 0;
mgraphics.autofill = 0;
var inlets = 3;
var outlets = 0;

// keyboard and piano roll parameters
var keyboardWidth = 50;
var pianoMin = 21;
var pianoMax = 108;
var numKeys = pianoMax - pianoMin + 1;
var keyHeight = 0; // will be computed as height/numKeys

// state variables
var notes = []; // stores note objects: {start, end, pitch, velocity}
var tempo = 1.0; // scroll speed multiplier; 1 means 10 sec roll width
var clockStart = null; // internal clock start time (in seconds)
var transitionTime = null; // absolute time for transition line
var transitionActive = false;

/**
 * Main paint function called by mgraphics.
 */
function paint() {
  var ctx = mgraphics;
  var width = ctx.size[0];
  var height = ctx.size[1];

  // compute key height for keyboard drawing
  keyHeight = height / numKeys;

  // clear background
  ctx.set_source_rgba(0.9, 0.9, 0.9, 1);
  ctx.rectangle(0, 0, width, height);
  ctx.fill();

  // draw piano keyboard on left side
  drawKeyboard(ctx, keyboardWidth, height);

  // define piano roll area dimensions
  var rollX = keyboardWidth;
  var rollWidth = width - keyboardWidth;

  // calculate effective time based on internal clock and tempo
  var now = (new Date()).getTime() / 1000.0;
  var effectiveTime = clockStart !== null ? (now - clockStart) * tempo : 0;

  // draw each note in the roll area
  ctx.set_line_width(1);
  for (var i = 0; i < notes.length; i++) {
    var note = notes[i];

    // skip note if it ended in the past relative to effective time
    if (note.end < effectiveTime) {
      continue;
    }

    // compute x positions for note start and end within the roll area
    var noteStartX = rollX + rollWidth * ((note.start - effectiveTime) / 10);
    var noteEndX = rollX + rollWidth * ((note.end - effectiveTime) / 10);

    // clip note if it starts before roll area
    if (noteStartX < rollX) {
      noteStartX = rollX;
    }
    // clip note if it extends past the roll area
    if (noteEndX > rollX + rollWidth) {
      noteEndX = rollX + rollWidth;
    }
    var noteWidth = noteEndX - noteStartX;

    // compute vertical position: higher pitches at the top
    var noteY = (pianoMax - note.pitch) * keyHeight;

    // draw the note rectangle
    ctx.set_source_rgba(0.2, 0.6, 0.8, 1);
    ctx.rectangle(noteStartX, noteY, noteWidth, keyHeight);
    ctx.fill();
    ctx.set_source_rgba(0, 0, 0, 1);
    ctx.rectangle(noteStartX, noteY, noteWidth, keyHeight);
    ctx.stroke();
  }

  // draw transition line if active
  if (transitionActive && transitionTime !== null) {
    var transX = rollX + rollWidth * ((transitionTime - effectiveTime) / 10);
    if (transX >= rollX && transX <= (rollX + rollWidth)) {
      ctx.set_source_rgba(1, 0, 0, 1);
      ctx.set_line_width(2);
      ctx.move_to(transX, 0);
      ctx.line_to(transX, height);
      ctx.stroke();
    }
  }
}

/**
 * Draws the piano keyboard on the left side.
 * @param {object} ctx - the mgraphics context
 * @param {number} kbWidth - width of the keyboard area
 * @param {number} height - height of the canvas
 */
function drawKeyboard(ctx, kbWidth, height) {
  // draw background for keyboard area
  ctx.set_source_rgba(1, 1, 1, 1);
  ctx.rectangle(0, 0, kbWidth, height);
  ctx.fill();

  // draw white key boundaries
  for (var p = pianoMin; p <= pianoMax; p++) {
    var y = (pianoMax - p) * keyHeight;
    if (isWhiteKey(p)) {
      ctx.set_source_rgba(1, 1, 1, 1);
      ctx.rectangle(0, y, kbWidth, keyHeight);
      ctx.fill();
      ctx.set_source_rgba(0, 0, 0, 1);
      ctx.move_to(0, y);
      ctx.line_to(kbWidth, y);
      ctx.stroke();
    }
  }
  // draw bottom boundary line
  ctx.move_to(0, height);
  ctx.line_to(kbWidth, height);
  ctx.stroke();

  // draw black keys (smaller rectangles)
  var blackKeyWidth = kbWidth * 0.6;
  var blackKeyHeight = keyHeight * 0.6;
  for (var p = pianoMin; p <= pianoMax; p++) {
    if (!isWhiteKey(p)) {
      var y = (pianoMax - p) * keyHeight + (keyHeight - blackKeyHeight);
      var x = (kbWidth - blackKeyWidth) / 2;
      ctx.set_source_rgba(0, 0, 0, 1);
      ctx.rectangle(x, y, blackKeyWidth, blackKeyHeight);
      ctx.fill();
    }
  }
}

/**
 * Returns true if the given MIDI pitch corresponds to a white key.
 * @param {number} pitch - MIDI pitch number
 * @return {boolean} true if white key, false if black key
 */
function isWhiteKey(pitch) {
  var mod = pitch % 12;
  return (mod === 0 || mod === 2 || mod === 4 || mod === 5 || mod === 7 || mod === 9 || mod === 11);
}

/**
 * Handles integer messages.
 * Inlet 0: if a single number, set clock start time.
 * Inlet 1: update tempo.
 * Inlet 2: set transition time.
 * @param {number} v - incoming integer
 */
function msg_int(v) {
  if (inlet === 0) {
    // if clockStart is not set and message is a single number, treat it as clock start
    if (clockStart === null) {
      clockStart = v;
    }
  } else if (inlet === 1) {
    // update tempo from inlet 1
    tempo = v;
  } else if (inlet === 2) {
    // set transition time from inlet 2 and activate transition line
    transitionTime = v;
    transitionActive = true;
  }
  mgraphics.redraw();
}

/**
 * Handles list messages.
 * Expects a list of 5 numbers: (start_time, end_time, pitch, velocity, sim)
 * Ignores the sim field.
 */
function list() {
  var args = arrayfromargs(messagename, arguments);
  if (inlet === 0 && args.length === 5) {
    var note = {
      start: args[0],
      end: args[1],
      pitch: args[2],
      velocity: args[3]
    };
    notes.push(note);
  }
  mgraphics.redraw();
}

mgraphics.redraw();