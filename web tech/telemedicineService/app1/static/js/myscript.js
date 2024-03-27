document.addEventListener('DOMContentLoaded', function () {
    // Initialize timeoutId outside the updateContent function
    let timeoutId = null;

    // Function to update content
    function updateContent() {
        // Create the checkmark icon
        const myicon = document.createElement('i');
        myicon.className = 'fa-solid fa-check';

        // Get the parent element (.modal-body)
        const modalBody = document.querySelector('.modal-body');

        // Create a new paragraph element
        const newParagraph = document.createElement('p');

        // Append the checkmark icon to the new paragraph
        newParagraph.appendChild(myicon);

        // Append the existing content to the new paragraph
        newParagraph.innerHTML += ' Huawei Watch GT 3';

        // Replace the existing paragraph with the new one
        modalBody.innerHTML = '';
        modalBody.appendChild(newParagraph);

        // Hide the spinner
        document.getElementById('myspan').style.display = 'none';
    }

    // Event listener for the button click
    document.getElementById('iotbutton2').addEventListener('click', function () {
        // Clear existing timeout
        clearTimeout(timeoutId);

        // Start the update after 4 seconds
        timeoutId = setTimeout(updateContent, 4000);
    });
});


// document.addEventListener('DOMContentLoaded', function() {
//   let timeoutId = null;

//   function updateContent() {
//     // Remove the spinner and add the checkmark
//     document.getElementById('myspan').style.display = 'none';
//     document.getElementById('myspan2').style.display = 'none';
//     document.getElementById('myspan3').style.display = 'none';
//     const myicon = document.createElement('i');
//     myicon.className = 'fa-solid fa-check myicon';
//     document.querySelector('.modal-body p').appendChild(myicon);
//   }

//   // Start the update after 4 seconds
//   if (timeoutId === null) {
//     timeoutId = setTimeout(updateContent, 7000);
//   }
// });


document.getElementById('iotbutton1').addEventListener('click', function (event) {
    event.preventDefault();
    document.getElementById('bloodpressure').value = '130–139/80–90 mm Hg';
    document.getElementById('heartrate').value = '75 to 115 bpm';
    document.getElementById('stresslevel').value = '76–100';
    document.getElementById('iotbutton2').textContent = 'Connected Successfully';
    document.getElementById('iotbutton2').className = 'btn btn-success btn-sm mybtn-success';
});


// document.getElementById('iotbutton').addEventListener('click', function (event) {
//     event.preventDefault(); // Prevent form submission and page reload
//     alert("Do you want to connect with IOT device?");
//     document.getElementById('bloodpressure').value = '130–139/80–90 mm Hg';
//     document.getElementById('heartrate').value = '75 to 115 bpm';
//     document.getElementById('stresslevel').value = '76–100';
//     document.getElementById('iotbutton').textContent = 'Connected Successfully';
//     document.getElementById('iotbutton').className = 'btn btn-success btn-sm';
// });

