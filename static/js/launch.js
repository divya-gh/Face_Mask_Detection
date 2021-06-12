 
// Render Images to select upon Launch
//-------------------------------------//

// Grab a reference to the dropdown select element
var selector = d3.select("#images");

// Make an API call to server to population the selection option with set of Images

d3.json("/api/v1.0/select_option").then((data) => { 
    var samples = data ;

    console.log(samples);
    data.forEach(file => {
        selector.append("option")
                .attr("value", file)
                .text(file)
    });

});


//On selection of a file, call the function renderImage to render the image for processing   
d3.select("#images").on("change", renderImage) ;

// Function renderImage 

function renderImage() {
    //Get options selected by the user
    var file = d3.selectAll('#images').node().value ;
    console.log(file);

    // Render selected Image for prediction
    d3.json(`/get_image/${file}`).then((data) => { 

        console.log(data);

    });

};

//When user clicks an option, add those options to div

$(document).ready(function(){
    $("a#live_Image").trigger("click");
});

$(document).on("click", "#live_Image", function(){
    console.log($(this).attr("id"));
    var selectedOption = $(this).attr("id");
    renderOption(selectedOption);
});

var liveImageHtml = '<h3>Live Image</h3> <div id="seconds-counter"> </div> <div id = "livevideo" class="col-lg-8  offset-lg-2">     <h3 class="mt-5">Live Streaming</h3>     <img src="{{ url_for("video_feed") }}" width="100%">     <a id= "takepicture" href="/capture_img" class="button primary">Take Picture  <i class="fas fa-camera"></i></a> </div>'

// $("#upload_Image").click(renderOption(this));
// $("#browse_Image").click(renderOption(this));
// $("#live_Image").click(renderOption(this));
// $("#analysis").click(renderOption(this));

function renderOption(selectedOption) {
    console.log(selectedOption);
    console.log(`#${selectedOption}`);
    if (selectedOption == "live_Image" ) {
        $(`#${selectedOption}`).html(liveImageHtml);
        var flaskPath = "/video_feed"
        callFlaskPath(flaskPath)
        var seconds = 30;
        var el = document.getElementById('seconds-counter');
        setInterval(decrementSeconds, 1000);
 
};

// https://stackoverflow.com/questions/37187504/javascript-second-counter
function decrementSeconds(el) {
    seconds -= 1;
    el.innerText = "Analysis will run in " + seconds + " seconds.";
}
    };

function callFlaskPath(path){
    d3.json(flaskPath).then((data) => { 

        console.log(data);
    });
};
