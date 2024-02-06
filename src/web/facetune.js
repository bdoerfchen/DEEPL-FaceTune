function getModelImgs() {
    return [
        {
            name: "model1",
            element: document.getElementById('model1-img')
        },
        {
            name: "model2",
            element: document.getElementById('model2-img')
        },
        {
            name: "model3",
            element: document.getElementById('model3-img')
        },
        {
            name: "model4",
            element: document.getElementById('model4-img')
        },
    ]
}

// Update image on preview
document.getElementById('upload-img').addEventListener('change', function() {
    var file = this.files[0];
    if (file) {
        var reader = new FileReader();
        reader.onload = async function(event) {
            image = event.target.result;
            document.getElementById('preview-img').setAttribute('src', image);
            //Upload image to python here
            r = await eel.decodeImage(image + "")();
            console.log(r)
            targets = getModelImgs();
            for (let i = 0; i < 4; i++)
            {
                targets[i].element.setAttribute('src', r[i])
            }

        }
        reader.readAsDataURL(file);
    }
});