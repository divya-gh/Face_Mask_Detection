//===============================================================================//
// Function to Initialize launch page
//===============================================================================//

function init() {
    //Ensure result is displayed 
    d3.select("#predict").style("display","block");

    //Make analysis report invisible
    d3.select("#analyze").style("display","none");

    // Disable camera
    d3.select("#livevideo").style("display","none");
      


}

// call init function
init();

//===================================================================================//
//    Create events for the SELECTION option on the main page
//==================================================================================//


// JQery - trigger events for links

$(document).ready(function(){
    $("a").trigger("click");

    // Function to render Predicted Image
    //===================================//

    renderPredictedImage = (file) => {
                    // Render selected Image for prediction
                    d3.json(`/get_image/${file}`).then((data) => { 
                        console.log("Prediction Data:", data);
        
                        var image_selector = d3.select("#predict_image")
                        image_selector.attr("src" , `${data.Image_path}`)
        
                    });
            }
    

    $(document).on("click", "#select", function(){
        console.log("You clicked the option:", $(this).attr("id"));

        //Ensure result is displayed 
        d3.select("#predict").style("display","block");

        //Make analysis report invisible
         d3.select("#analyze").style("display","none");

        // Disable camera
        d3.select("#livevideo").style("display","none");

        //Get deafault image prediction
        // renderPredictedImage("people1.jpg");
        var image_selector = d3.select("#predict_image")
            image_selector.attr("src" , `./static/images/default_prediction.jpg`)

        // Make an API call to server to population the selection option with set of Images
        d3.json("/api/v1.0/select_option").then((data) => { 
            var samples = data ;

            console.log("data for creating radio buttons:", samples);

            createRadioButtons(data);

        });

    // D3 to create radio buttons
    function createRadioButtons(data) {
        //clear previous svg data
        d3.select("div#select_option").html("")

        // set class
        var selection = d3.select("#select_option").classed("set_div_height", true)

        var form = selection.append('form');

        radiogroup = form.selectAll("radios").data(data).enter()
            .append('g').classed("radios" , true) ;

        // Append input elements and text to each radioGroup
        var j=0;
        radiogroup.append('input')
                  .attr('type', 'radio')
                  .attr('value', d => d)
                  .attr('name', 'toggle')
                //   .on('click', function () {
                //       //Do something
                //   });
                .property("checked", function(d, i) { 
                    return (i===j); 
                })

        // Assign labels to radio button group
        radiogroup.append('label')
                  .text(d =>d)
                  .style("padding", "3px");

        //------------------------------------------------------------//
        //Event Handlling with Radio buttons//
        //-----------------------------------------------------------//

        //When a button is clicked,
        d3.selectAll("input").on("click" , () => {

            // get selected file
            var value = d3.select('input[name="toggle"]:checked').node().value
            console.log(value)

            renderPredictedImage(value)
        }) ;
        // End of Radiobutton click
    } //end of createRadioButtons function

    }); //End of Selection on click
    


//===================================================================================//
//    Create events for the BROWSE option on the main page
//==================================================================================//
        $(document).on("click", "#browse", function(){
            //print clicked option
            console.log("You clicked the option:", $(this).attr("id"));
            //Ensure result is displayed 
            d3.select("#predict").style("display","block");

            // Disable camera
            d3.select("#livevideo").style("display","none");

            //Make analysis report invisible
            d3.select("#analyze").style("display","none");

            //clear previous row data
            d3.select("div#select_option").html("")

            // set class
            var selection = d3.select("#select_option").classed("set_browse_height", true).classed("set_div_height", false)
            
            // // set input form
            var inputForm = '<label for="formFileSm" class="form-label h6 m-0">Upload a picture with maximum neighbours = 3 <br><span style="font-size:13px;"> ( File Format: .jpg, .png )</span> </label><input class="form-control form-control-sm set-browse rounded-lg w-50" id="myFile" type="file" /><h6 id="note" style="font-size:13px;">Note: Save your pictures in the<em>UploadPic</em> folder.</h6>'
            var browseForm = selection.html(inputForm)
            
            //enent handing : get value on file upload
            browseForm.on("change" , () => {
                console.log("get browsed file:", d3.event.target.value)
                // get uploaded file path
                var file = d3.event.target.value ;                

                console.log("Uploaded file:", file.split("\\")[2])

                var uploaded_File = file.split("\\")[2] ;

                // get prediction only if file name is not undefined
                if (uploaded_File != undefined) {

                    renderPredictedImage(uploaded_File);
                }

            // --------- TODO Copy to Main ------------- //
            // var inputForm = '<form enctype="multipart/form-data" action="/upload" method="post"> \
            //                      <div class="form-group"> \
            //                      <label for="formFileSm" class="form-label h6 m-0">Upload a picture with maximum neighbours = 3 <br>\
            //                      <span style="font-size:13px;"> ( File Format: .jpg, .png )</span> </label>\
            //                      <input class="form-control form-control-sm set-browse rounded-lg w-50" id="myFile" type="file" />\
            //                      <br>\
            //                      <button type = "submit" class="btn btn-success">Submit</button> \
            //                      </div> \
            //                      <h6 id="upload-message" class="text-primary"><strong></strong></h6> \
            //                 </form>'
            // --------- TODO Copy to Main ------------- //
            // var browseForm = selection.html(inputForm)

            // //enent handing : get value on file upload
            // browseForm.on("submit" , () => {
            //     // --------- TODO Copy to Main ------------- //
            //     console.log("get browsed file:", d3.event.target.value)
            //     d3.select("#note").text("Image uploaded successfully!")
            //     var saveLocation = './static/upload'
            //     var fileFakePath = $("#myFile").val()
            //     var fileName = fileFakePath.substring(12)

            //     var runPath = `${saveLocation}/${fileName}`
            //     console.log(runPath)

            //     // renderPredictedImage(runPath)
            //     // Render selected Image for prediction
            //     d3.json(`/get_image/${fileName}`).then((data) => { 
            //         console.log("Prediction Data:", data);
    
            //         var image_selector = d3.select("#predict_image")
            //         image_selector.attr("src" , `${data.Image_path}`)
    
            //     });
                // --------- TODO Copy to Main ------------- //




            });  //End of browForm on change

        });//End of Browse selection option




    //===================================================================================//
    //    Create events for the CAMERA option on the main page
    //==================================================================================//

    $(document).on("click", "#camera", function(){
        //print clicked option
        console.log("You clicked the option:", $(this).attr("id"));

        d3.select("#livevideo").style("display","block");

        //Ensure result is displayed 
        d3.select("#predict").style("display","none");

        //Make analysis report invisible
        d3.select("#analyze").style("display","none");

        //clear previous row data
        d3.select("div#select_option").html("").classed("set_browse_height set_div_height" , false)

        // var camera_selector= d3.select("#show_camera");
        // camera_selector.attr("src", "{{ url_for('video_feed') | safe }}")
        // console.log(camera_selector.attr('src'))




    });




//===================================================================================//
//    Create events for the ANALYSIS option on the main page
//==================================================================================//
    $(document).on("click", "#analysis", function(){
        //print clicked option
        console.log("You clicked the option:", $(this).attr("id"));

        // Disable camera
        d3.select("#livevideo").style("display","none");

        //Make analysis report visible
        d3.select("#analyze").style("display","block");
        
        //clear previous row data
        d3.select("div#select_option").html("")

        // set class false
        d3.select("div#select_option").html("").classed("set_browse_height set_div_height" , false)

        //Make prediction result invisible
        d3.select("#predict").style("display","none");
        
        // Add anaysis.html to div container
        // document.getElementById("analyze").innerHTML='<object type="text/html" data="./static/analysis.html" ></object>';
        $("#analyze").load("./static/analysis.html");

    }); // End of click analysis


    //===================================================================================//
//    Create events for the Find Us option on the main page
//==================================================================================//
// $(document).on("click", "#analysis", function(){
//     //print clicked option
//     console.log("You clicked the option:", $(this).attr("id"));

//     // Disable camera
//     d3.select("#livevideo").style("display","none");

//     //Make analysis report visible
//     d3.select("#analyze").style("display","block");
    
//     //clear previous row data
//     d3.select("div#select_option").html("")

//     // set class false
//     d3.select("div#select_option").html("").classed("set_browse_height set_div_height" , false)

//     //Make prediction result invisible
//     d3.select("#predict").style("display","none");
    
//     // Add anaysis.html to div container
//     // document.getElementById("analyze").innerHTML='<object type="text/html" data="./static/analysis.html" ></object>';
//     $("#analyze").load("./static/analysis.html");

// }); // End of click analysis



});//end of Jquery










