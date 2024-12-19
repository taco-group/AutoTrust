function enable_submit(param) {
    $('#placeholder').text(param);
    $('#placeholder').removeClass("placeholder-init");
    $('#placeholder').addClass("placeholder");
  }
  
var show = false;  

function set_border(elements) {
    for (var i = 0; i < elements.length; i++) {
        elements[i].style.border = "2px solid #04af58";
    }
}

function unset_border(elements) {
    for (var i = 0; i < elements.length; i++) {
        elements[i].style.border = "none";
    }
}

function initialize() {
  var trigger = $('#hint');
  show = false;
  $('#tips').text('(Click to show the answers)');
  trigger.click(function () {
    var elements = document.getElementsByClassName("correct-answer");  
    if (show) {
        show = false;
        unset_border(elements);
        $('#tips').text('(Click to show the answers)');
    } else {
        show = true;
        set_border(elements);
        $('#tips').text('(Click to hide the answers)');
    }
  });
}

$(document).ready(function() {
  initialize();
});