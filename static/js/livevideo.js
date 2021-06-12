// imgCapButton = d3.getElementbyId('#takepicture');
// video = d3.getElementbyId('#livevideo');

// imgCapButton.onclick = hideVideo(video);

// function hideVideo(video) {
//     video.style("visibility", "hidden");
// };

//When user clicks an option, add those options to div

$(document).ready(function(){
    $("a#live_Image").trigger("click");
});

$(document).on("click", "#live_Image", function(){
    console.log($(this).attr("id"));
    var selectedOption = $(this).attr("id");
    renderOption(selectedOption, callFlaskPath);
});

var liveImageHtml = `<div id = "livevideo" class="col-lg-8  offset-lg-2"> <h3 class="mt-5">Live Streaming</h3> <img src="{{ url_for('video_feed') }}" width="100%"> </div>`

// $("#upload_Image").click(renderOption(this));
// $("#browse_Image").click(renderOption(this));
// $("#live_Image").click(renderOption(this));
// $("#analysis").click(renderOption(this));

function renderOption(selectedOption, callFlaskPath) {
    console.log(selectedOption);
    console.log(`#${selectedOption}`);
    if (selectedOption == "live_Image" ) {
        $('#collapse-video').collapse({
            toggle: true
          })
        $('#collapse-browse').collapse({
            toggle: false
          })
        $('#collapse-analysis').collapse({
            toggle: false
          })
        $('#collapse-find-us').collapse({
            toggle: false
          })       
        $('#collapse-upload').collapse({
            toggle: false
          })                  
        // let flaskPath = "/video_feed";

        // $("#selected-option").append(liveImageHtml);
        // callFlaskPath(flaskPath);
        // // buildDiv(liveImageHtml, flaskPath, callFlaskPath);
        // // var seconds = 30;
        // // var el = document.getElementById('seconds-counter');
        // // setInterval(decrementSeconds, 1000);
    };
};

// https://stackoverflow.com/questions/37187504/javascript-second-counter
// function decrementSeconds(el) {
//     seconds -= 1;
//     el.innerText = "Analysis will run in " + seconds + " seconds.";
// }
//     };

function callFlaskPath(path){
    d3.json(path).then((data) => { 
        // console.log(data);
    });
};

// function buildDiv(html, path, callFlaskPath){
//     if (html){
//         document.getElementById("#selected-option").innerHtml=html;
//     };
//     callFlaskPath(path);
// };