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


function setLatentSpace(file, model) {
   var reader = new FileReader();
   reader.onload = async function(event) {
       latent = event.target.result;
       //Upload image to python here
       r = await eel.decodeLatent(latent + "")();
       console.log(r)
       targets = getModelImgs()[model - 1].element.setAttribute('src', r);
    }
   reader.readAsDataURL(file);
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

document.getElementById('upload-latent1').addEventListener('change', function() {
    var file = this.files[0];
    if (file) {
        setLatentSpace(file,1)
    }
});

document.getElementById('upload-latent2').addEventListener('change', function() {
    var file = this.files[0];
    if (file) {
        setLatentSpace(file,2)
    }
});

document.getElementById('upload-latent3').addEventListener('change', function() {
    var file = this.files[0];
    if (file) {
        setLatentSpace(file,3)
    }
});

document.getElementById('upload-latent4').addEventListener('change', function() {
    var file = this.files[0];
    if (file) {
        setLatentSpace(file,4)
    }
});