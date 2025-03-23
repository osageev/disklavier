mgraphics.init();
mgraphics.relative_coords = 0;
mgraphics.autofill = 0;

var width;
var height;
var diameter = 20;
var isPlaying = false;
var cursor = 0;
var max_length = 60000;
var startTime = 0;

var color1 = [0.2, 0.2, 0.2, 1];
var color2 = [1, 0.5, 0.5, 1];

var t = new Task(cursorRun, this);

function paint() {
  width = this.box.rect[2] - this.box.rect[0];
  height = this.box.rect[3] - this.box.rect[1];

  with (mgraphics) {
    // background
    set_source_rgba(color1);
    rectangle(0, 0, width, height);
    fill();

    // play/pause button background
    set_source_rgb(color2);
    ellipse(
      width / 2 - diameter / 2,
      height / 2 - diameter / 2,
      diameter,
      diameter
    );
    fill();

    // play/pause symbol
    set_source_rgba(color1);
    if (!isPlaying) {
      move_to(width / 2 - 3.5, height / 2 - 8);
      line_to(width / 2 - 3.5, height / 2 + 8);
      line_to(width / 2 + 5.5, height / 2);
      close_path();
      fill();
    } else {
      rectangle(width / 2 - 4.5, height / 2 - 5, 3, 10);
      fill();
      rectangle(width / 2 + 1.5, height / 2 - 5, 3, 10);
      fill();
    }

    // time display
    select_font_face("Arial Black");
    set_font_size(16);
    set_source_rgb(color2);
    move_to(15, height / 2 + 5);
    text_path(fancyTimeFormat(cursor * 0.001));
    fill();
  }
}

function fancyTimeFormat(duration) {
  // hours, minutes, and seconds
  const hrs = Math.floor(duration / 3600);
  const mins = Math.floor((duration % 3600) / 60);
  const secs = Math.floor(duration % 60);
  const decs = Math.floor((duration - Math.floor(duration)) * 1000);

  // output like "1:01" or "4:03:59" or "123:03:59"
  let ret = "";

  ret += "" + (hrs ? hrs : 0) + ":" + (mins < 10 ? "0" : "");
  ret += "" + mins + ":" + (secs < 10 ? "0" : "");
  ret += "" + secs;

  return ret;
}

function set_cursor(c) {
  cursor = c;
  if (cursor > max_length) set_max_length(cursor);
  mgraphics.redraw();
}

function set_max_length(length) {
  max_length = length;
  mgraphics.redraw();
}

function cursorRun() {
  const currentTime = new Date().getTime();
  const elapsedTime = currentTime - startTime;
  set_cursor(elapsedTime);
  if (cursor >= max_length) {
    t.cancel();
    isPlaying = false;
  }
}

function onclick(x, y, but, cmd, shift, capslock, option, ctrl) {
  if (
    y > height / 2 - diameter / 2 &&
    y < height / 2 + diameter / 2 &&
    x > width / 2 - diameter / 2 &&
    x < width / 2 + diameter / 2
  ) {
    if (!isPlaying) {
      isPlaying = true;
      startTime = new Date().getTime() - cursor;
      t.interval = 10; // every 10 milliseconds
      t.repeat();
    } else {
      isPlaying = false;
      t.cancel();
    }
    mgraphics.redraw();
  }
}

function init() {
  this.box.size(200, 100);
}

init();
mgraphics.redraw();
