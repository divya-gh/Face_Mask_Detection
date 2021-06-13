//===================================================================================//
//    Create events for the selection option on the main page
//==================================================================================//


// JQery - trigger events for links

$(document).ready(function(){
    $("a").trigger("click");
    

    $(document).on("click", "#select", function(){
        console.log($(this).attr("id"));

        // Make an API call to server to population the selection option with set of Images

    d3.json("/api/v1.0/select_option").then((data) => { 
        var samples = data ;

        console.log(samples);

        createRadioButtons(data);

});

// D3 to create radio buttons
function createRadioButtons(data) {
    //clear previous svg data
    d3.select("div#select_option").html("")

    // Experiment
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
        var value = d3.select('input[name="toggle"]:checked').node().value
        console.log(value)

        // Render selected Image for prediction
        d3.json(`/get_image/${value}`).then((data) => { 
            console.log(data);

        });
        }) ;
    // End of button click
        } //end of createRadioButtons function

        });
    });//end of Jquery







