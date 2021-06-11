 
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

}