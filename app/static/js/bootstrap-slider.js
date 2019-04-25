// Without JQuery
var slider = new Slider("#ex6");
slider.on("slide", function(sliderValue) {
	document.getElementById("ex6SliderVal").textContent = sliderValue;
});
