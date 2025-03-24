// velocity.js - a MaxMSP visualization for MIDI note velocities
// displays a scrolling window of velocity bars with real-time updates

autowatch = 1;

// configuration parameters
var config = {
    maxNotes: 100,        // maximum number of notes to display
    timeWindow: 1000,     // time window in ms for "recent" notes (1 second)
    bgColor: [0.1, 0.1, 0.1, 1.0],
    recentColor: [0.2, 0.8, 0.2, 1.0],  // green for recent notes
    oldColor: [0.2, 0.4, 0.8, 1.0],     // blue for older notes
    textColor: [1.0, 1.0, 1.0, 1.0],
    barWidth: 5,
    barGap: 2
};

// data storage
var noteHistory = [];  // will store {velocity, timestamp} objects

/**
 * initialize the sketch object
 */
function setup() {
    // create a sketch object for drawing
    var sketch = new JitterObject("jit.gl.sketch", "v8ui");
    sketch.automatic = 0;
    
    // create an interval to update the display regularly
    var interval = new Task(function() {
        draw(sketch);
    }, this);
    interval.interval = 33; // ~30fps refresh rate
    interval.repeat();
    
    return sketch;
}

var sketch = setup();

/**
 * handle incoming note velocity
 * 
 * @param {number} velocity - MIDI velocity value (0-127)
 */
function msg_int(velocity) {
    // add new note to history with current timestamp
    noteHistory.push({
        velocity: velocity,
        timestamp: Date.now()
    });
    
    // remove old notes if we have too many
    if (noteHistory.length > config.maxNotes) {
        noteHistory.shift();
    }
    
    // we don't need to explicitly call draw here since we're using
    // an interval task to refresh the display
}

/**
 * draw the visualization
 * 
 * @param {object} sketch - jit.gl.sketch object for drawing
 */
function draw(sketch) {
    var now = Date.now();
    var width = sketch.glparent.drawto.size[0];
    var height = sketch.glparent.drawto.size[1];
    
    // clear the background
    sketch.glclearcolor = config.bgColor;
    sketch.glclear();
    
    // calculate recent average (last second)
    var recentNotes = noteHistory.filter(function(note) {
        return now - note.timestamp < config.timeWindow;
    });
    
    var avgVelocity = 0;
    if (recentNotes.length > 0) {
        var sum = recentNotes.reduce(function(acc, note) {
            return acc + note.velocity;
        }, 0);
        avgVelocity = Math.round(sum / recentNotes.length);
    }
    
    //
    // draw note history as vertical bars
    var totalBarWidth = config.barWidth + config.barGap;
    var maxBars = Math.min(noteHistory.length, Math.floor(width / totalBarWidth));
    
    for (var i = 0; i < maxBars; i++) {
        var note = noteHistory[noteHistory.length - 1 - i];
        var isRecent = (now - note.timestamp < config.timeWindow);
        
        // normalize velocity to drawing height (0-127 to 0-height)
        var barHeight = (note.velocity / 127.0) * (height * 0.8);
        
        // position bar
        var x = width - (i + 1) * totalBarWidth;
        var y = height - barHeight;
        
        // set color based on recency
        sketch.glcolor = isRecent ? config.recentColor : config.oldColor;
        
        // draw bar
        sketch.shapeslice(4, 4);
        sketch.rectangle(x, y, config.barWidth, barHeight);
    }
    
    // draw average velocity text in upper left
    sketch.glcolor = config.textColor;
    sketch.textalign("left", "top");
    sketch.fontsize(14);
    sketch.text("Avg: " + avgVelocity, 10, 10);
    
    // render the sketch
    sketch.render();
}

/**
 * resize handler for the UI
 */
function resize() {
    if (sketch) {
        draw(sketch);
    }
}

/**
 * cleanup function when script is recompiled or deleted
 */
function clean() {
    if (sketch) {
        sketch.freepeer();
    }
}
