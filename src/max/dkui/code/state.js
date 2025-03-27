var outlets = 3;

var sim = 0;
var tempo = 60;

function setTempo(t) {
  post("tempo from ", tempo, "to", t);
  tempo = t;
  outlet(0, "tempo", tempo);
}

function getTempo() {
  outlet(0, "tempo", tempo);
}

function setSim(s) {
  post("sim from ", sim, "to", s);
  sim = s;
  outlet(1, "sim", sim);
}

function getSim() {
  outlet(1, "sim", sim);
}
