/* Files that handle interaction with the predictions form
This file must be included whenever the form is used. */

updateFileNames = function(remove) {
    var input = document.getElementById('geometry_file');
    var output = document.getElementById('uploaded_file');

    if (remove || input.files.length == 0) {
        output.textContent = "No file selected";
    } else {
        output.textContent= "Uploaded file: " + input.files.item(0).name;
    }
    
}

fileDrop = function(event) {
    event.preventDefault();
    var input = document.getElementById('geometry_file');
    input.files = event.dataTransfer.files;
    event.target.classList.remove("border-teal-600");
    updateFileNames(false);
}

fileDrag = function(event) {
    event.preventDefault();
    event.target.classList.add("border-teal-600");
}

fileDragLeave = function(event) {
    event.preventDefault();
    event.target.classList.remove("border-teal-600");
}

formReset = function() {
    updateFileNames(true);
    document.getElementById("loading").classList.add("hidden");
    document.getElementById("error").classList.add("hidden");
}

formSubmit = function(event) {
    event.preventDefault();

    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("error").classList.add("hidden");
    document.getElementById("submit_button").disabled = true;

    // TODO do something here to show user that form is being submitted
    fetch(event.target.action, {
        method: event.target.method,
        body: new FormData(event.target) // event.target is the form
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