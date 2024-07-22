/* Files that handle interaction with the predictions form
This file must be included whenever the form is used. */

updateFileNames = function(input, output, remove) {

    if (remove || input.files.length == 0) {
        output.textContent = "No file selected";
    } else {
        output.textContent= "Uploaded file: " + input.files.item(0).name;
    }

}

fileInputChange = function(event) {
    var input = event.target;
    var output = input.closest(".file-drop-container").querySelectorAll(".uploaded-files")[0];

    updateFileNames(input, output);
}

fileDrop = function(event) {
    event.preventDefault();
    var input = event.target.querySelectorAll("input[type=file]")[0];
    var output = event.target.closest(".file-drop-container").querySelectorAll(".uploaded-files")[0];

    input.files = event.dataTransfer.files;
    event.target.classList.remove("border-teal-600");
    updateFileNames(input, output, false);
}

fileDrag = function(event) {
    event.preventDefault();
    event.target.classList.add("border-teal-600");
}

fileDragLeave = function(event) {
    event.preventDefault();
    event.target.classList.remove("border-teal-600");
}

formReset = function(event) {
    form = event.target;

    //Update all the file inputs
    var file_inputs = form.querySelectorAll(".file-drop-container");
    file_inputs.forEach((input) => updateFileNames(input.querySelectorAll("input[type=file]")[0], input.querySelectorAll(".uploaded-files")[0], true));

    document.getElementById("loading").classList.add("hidden");
    document.getElementById("error").classList.add("hidden");
}

formSubmit = function(event) {
    event.preventDefault();

    form = event.target;

    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("error").classList.add("hidden");
    document.getElementById("submit_button").disabled = true;

    // TODO do something here to show user that form is being submitted
    fetch(form.action, {
        method: form.method,
        body: new FormData(form)
    }).then(async (response) => {

        document.getElementById("loading").classList.add("hidden");
        document.getElementById("submit_button").disabled = false;

        if (!response.ok) {

            var error_element = document.getElementById("error")
            var error_m_element = document.getElementById("error_message")
            error_m_element.textContent = `HTTP error! Status: ${response.status}`;
            error_element.classList.remove("hidden");

        }

        // Get the name of the received file
        var filename = response.headers.get("content-disposition").split("filename=")[1].split(";")[0].slice(1, -1)

        return {filename: filename, blob: await response.blob()};
    }).then(({filename, blob}) => {
        var file_url = window.URL.createObjectURL(blob);

        let link = document.createElement('a');
        link.href = file_url;
        link.download = filename;
        link.click();

    }).catch((error) => {
        // TODO handle error
        console.warn(error)
    });
}

formSubmitTest = function(event) {
    event.preventDefault();

    form = event.target;

    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("error").classList.add("hidden");
    document.getElementById("submit_button").disabled = true;

    // TODO do something here to show user that form is being submitted
    fetch(form.action, {
        method: form.method,
        body: new FormData(form)
    }).then(async (response) => {

        document.getElementById("loading").classList.add("hidden");
        document.getElementById("submit_button").disabled = false;

        if (!response.ok) {

            var error_element = document.getElementById("error")
            var error_m_element = document.getElementById("error_message")
            error_m_element.textContent = `HTTP error! Status: ${response.status}`;
            error_element.classList.remove("hidden");

        }

        return response.json();
    }).then((json) => {

        var pre_element = document.createElement("pre");
        pre_element.textContent = JSON.stringify(json, null, 2);

        output_div = document.getElementById("output_div")
        output_div.classList.remove("hidden");

        // Add pre tag to output div
        output_div.appendChild(pre_element);

    }).catch((error) => {
        // TODO handle error
        console.warn(error)
    });
}
