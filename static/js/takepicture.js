imgCapButton = d3.getElementbyId('#takepicture');
video = d3.getElementbyId('#livevideo');

imgCapButton.onclick = hideVideo(video);

function hideVideo(video) {
    video.style("visibility", "hidden");
};