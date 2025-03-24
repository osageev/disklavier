mgraphics.init();
mgraphics.relative_coords = 0;
mgraphics.autofill = 0;

var windowHeight;
var windowWidth;
var velocityHistory = []; // Store velocity events with timestamps
var averageVelocity = 0; // Current average velocity

function msg_int(velocity) {
  if (velocity > 0) {
    // Store velocity with current timestamp
    var now = new Date().getTime();
    velocityHistory.push({ velocity: velocity, timestamp: now });

    // Remove velocities older than 1 second
    var oneSecondAgo = now - 1000;
    velocityHistory = velocityHistory.filter(function (item) {
      return item.timestamp >= oneSecondAgo;
    });

    // Calculate average
    var sum = 0;
    var count = velocityHistory.length;

    for (var i = 0; i < count; i++) {
      sum += velocityHistory[i].velocity;
    }

    averageVelocity = count > 0 ? Math.round(sum / count) : 0;

    // Print calculation details
    // post("new velocity:", velocity, "\n");
    // post("velocities in last second:", count, "\n");
    // post("sum:", sum, "\n");
    // post("average:", averageVelocity, "\n");

    // Trigger redraw to update display
    mgraphics.redraw();
  }
}

function paint() {
  windowHeight = this.box.rect[2] - this.box.rect[0];
  windowWidth = this.box.rect[3] - this.box.rect[1];

  // Clear background
  mgraphics.set_source_rgba(0.15, 0.15, 0.15, 1.0);
  mgraphics.rectangle(0, 0, windowHeight, windowWidth);
  mgraphics.fill();

  // Display average velocity in large text
  mgraphics.set_source_rgba(1, 1, 1, 1);
  mgraphics.select_font_face("Arial Bold");

  // Calculate font size based on box dimensions
  var fontSize = Math.min(windowWidth * 0.5, windowHeight * 0.2);
  mgraphics.set_font_size(fontSize);

  // Center text
  var text = averageVelocity.toString();
  var textWidth = mgraphics.text_measure(text)[0];
  var textHeight = fontSize;

  mgraphics.move_to(
    (windowHeight - textWidth) / 2,
    (windowWidth + textHeight / 2) / 2
  );
  mgraphics.text_path(text);
  mgraphics.fill();

  //draw label
  fontSize = Math.min(windowWidth * 0.15, windowHeight * 0.08);
  mgraphics.set_font_size(fontSize);
  var label = "avg velocity";
  textWidth = mgraphics.text_measure(label)[0];

  mgraphics.move_to((windowHeight - textWidth) / 2, windowWidth * 0.85);
  mgraphics.text_path(label);
  mgraphics.fill();
}

function init() {
  this.box.size(160, 120);

  // Set initial dimensions
  windowHeight = this.box.rect[2] - this.box.rect[0];
  windowWidth = this.box.rect[3] - this.box.rect[1];
}

init();
mgraphics.redraw();
