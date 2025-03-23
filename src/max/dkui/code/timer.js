mgraphics.init();
mgraphics.relative_coords = 0;
mgraphics.autofill = 0;

var width;
var height;
var diameter = 20;
var isRunning = false;
var time = 0;
var targetTime = 60000; // default 60 seconds (in ms)
var startTime = 0;
var lastTick = 0;

var color1 = [0.2, 0.2, 0.2, 1]; // background color
var color2 = [1, 0.5, 0.5, 1]; // accent color

// initialize task for timer updates
t = new Task(timerTick, this);

/**
 * Format time in a nice readable format.
 *
 * Parameters
 * ----------
 * duration : number
 *     Time in milliseconds.
 *
 * Returns
 * -------
 * array
 *     Array containing formatted time string and decimal portion.
 */
function fancyTimeFormat(duration) {
  // hours, minutes and seconds
  const hrs = ~~(duration / 3600);
  const mins = ~~((duration % 3600) / 60);
  const secs = ~~duration % 60;
  const decs = (duration - Math.floor(duration)) * 1000;

  // output like "1:01" or "4:03:59"
  var ret = "";

  ret = ret + "" + (hrs ? hrs : 0) + ":" + (mins < 10 ? "0" : "");
  ret = ret + "" + mins + ":" + (secs < 10 ? "0" : "");
  ret = ret + "" + secs;

  var a = new Array();
  a[0] = ret;
  a[1] = decs;

  return a;
}

function init() {
  this.box.size(250, 70);
}

// function play() {
//   // background
//   mgraphics.set_source_rgb(color2);
//   mgraphics.ellipse(
//     width / 2 - diameter / 2,
//     20 - diameter / 2,
//     diameter,
//     diameter
//   );
//   mgraphics.fill();

//   // play symbol
//   mgraphics.set_source_rgba(color1);
//   mgraphics.move_to(width / 2 - 3.5, 12);
//   mgraphics.line_to(width / 2 - 3.5, 28);
//   mgraphics.line_to(width / 2 + 5.5, 20);
//   mgraphics.line_to(width / 2 - 3.5, 12);
//   mgraphics.close_path();
//   mgraphics.fill();
// }


function paint() {
  width = this.box.rect[2] - this.box.rect[0];
  height = this.box.rect[3] - this.box.rect[1];

  with (mgraphics) {
    // background
    set_source_rgba(color1);
    rectangle(0, 0, width, height);
    fill();

    // time display
    select_font_face("Arial Black");
    set_font_size(16);
    set_source_rgb(color2);
    move_to(15, 25);
    text_path(fancyTimeFormat(time * 0.001)[0]);
    fill();

    select_font_face("Arial");
    set_font_size(12);
    set_source_rgb(color2);
    move_to(15 + 68, 25);
    text_path(Math.floor(fancyTimeFormat(time * 0.001)[1]).toString());
    fill();

    // target time display
    select_font_face("Arial");
    set_font_size(12);
    set_source_rgb(color2);
    move_to(width - 80, 55);
    text_path("Target: " + fancyTimeFormat(targetTime * 0.001)[0]);
    fill();
  }
}

function timerTick() {
  var now = new Date().getTime();
  if (lastTick > 0) {
    time += now - lastTick;
  }
  lastTick = now;

  if (time >= targetTime) {
    time = targetTime;
    isRunning = false;
    t.cancel();
    outlet(0, "done");
  }

  mgraphics.redraw();
}

function set_time(ms) {
  time = ms;
  mgraphics.redraw();
}

function list(p, v) {
  // post("list", p, v);
  // post();
}

function bang() {
  if (!isRunning) {
    isRunning = true;
    if (time >= targetTime) {
      time = 0;
    }
    lastTick = new Date().getTime();
    t.interval = 10;
    t.repeat();
  } else {
    isRunning = false;
    t.cancel();
  }
  mgraphics.redraw();
}

// Initialize and draw UI
init();
mgraphics.redraw();
