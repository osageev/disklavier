mgraphics.init();
mgraphics.relative_coords = 0;
mgraphics.autofill = 0;

var width;
var height;
var cursor = 0;
var bgColor = [0.2, 0.2, 0.2, 1];
var color2 = [0.5, 1.0, 0.5, 1];

g = new Global("max_time");

var itemArray = [];

function Item() {
  this.object = "";
  this.elements = [];
}

function ItemElement(t, v, i) {}

function isNumber(n) {
  return !isNaN(parseFloat(n)) && !isNaN(n - 0);
}

function fancyTimeFormat(duration) {
  // Hours, minutes and seconds
  const hrs = ~~(duration / 3600);
  const mins = ~~((duration % 3600) / 60);
  const secs = ~~duration % 60;
  const decs = (duration - Math.floor(duration)) * 1000;

  var ret = "";

  ret = ret + "" + (hrs ? hrs : 0) + ":" + (mins < 10 ? "0" : "");

  ret = ret + "" + mins + ":" + (secs < 10 ? "0" : "");
  ret = ret + "" + secs;

  var a = new Array();

  a[0] = ret;
  a[1] = decs;

  return a;
}

function paint() {
  width = this.box.rect[2] - this.box.rect[0];
  height = this.box.rect[3] - this.box.rect[1];

  with (mgraphics) {
    // background
    set_source_rgba(bgColor);
    rectangle(0, 0, width, height);
    fill();

    // time
    select_font_face("Arial Black");
    set_font_size(16);
    set_source_rgb(color2);
    move_to(15, 25);
    text_path(fancyTimeFormat(cursor * 0.001)[0]);
    fill();

    select_font_face("Arial");
    set_font_size(12);
    set_source_rgb(color2);
    move_to(15 + 68, 25);
    text_path(Math.floor(fancyTimeFormat(cursor * 0.001)[1]).toString());
    fill();
  }
}

function set_cursor(c) {
  cursor = c;
  if (itemArr.length > 0) {
    for (var i = 0; i < itemArr.length; i++) {
      send_val = linearInterpolation(itemArr[i].elements, cursor);
      g.value = send_val;
      g.sendnamed(itemArr[i].object, "value");
    }
  }
  if (cursor > max_length) set_max_length(cursor);
  mgraphics.redraw();
}

function set_max_length(length) {
  max_length = length;
  mgraphics.redraw();
}

t = new Task(cursorRun, this);

function cursorRun() {
  if (cursor >= max_length) {
    t.cancel();
    isPlaying = false;
    return;
  }
  set_cursor(cursor + 10);
}

function init() {
  this.box.size(500, 80);
}

init();

set_cursor(
  Math.max(Math.min((x - 7.5) * 125 * (max_length / 60000), max_length), 0)
);

mgraphics.redraw();
