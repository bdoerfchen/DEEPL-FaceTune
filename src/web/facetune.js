pageMain = document.getElementById('page-main')
pageSlider = document.getElementById('page-slider')
currentLatent = Array(64).fill(0)
currentModelIndex = 0

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

// Update latent vector
document.getElementById('upload-latent').addEventListener('change', function() {
    var file = this.files[0];
    if (file) {
        var reader = new FileReader();
        reader.onload = async function(event) {
            latent_json = event.target.result;
            latent = JSON.parse(latent_json)
            buildSlider(latent)
            changeLatentParameter(-1);
        }
        reader.readAsText(file);
    }
});

function openLatent(open, index) {
    pageMain.classList.toggle('hidden')
    pageSlider.classList.toggle('hidden')

    if (open) {
        currentModelIndex = index
        img = document.getElementById('preview-img').src
        eel.getEncoding(currentModelIndex, img)().then(latent => {
            console.log("latent", latent)
            buildSlider(latent);
            changeLatentParameter(-1);
        })

    }
}

async function changeLatentParameter(index)
{
    decodedTarget = document.getElementById('latent-output-img')

    console.log(index)

    if (index > 0)
    {
        value = document.getElementById("slider" + index).value;
        currentLatent[index] = value;
    }
    //Get new image
    console.log(currentLatent)
    img = await eel.decodeLatentEncoding(currentModelIndex, currentLatent)();
    decodedTarget.src = img;
    updateDownloadLink();
}

function buildSlider(latent) {
    currentLatent = latent

    lockSlider = currentModelIndex == 1

    sliderContainer = document.getElementById("slider-container")
    sliderContainer.innerHTML = "";
    for (let i = 0; i < 64; i++)
    {
        newSlider = document.createElement('input')
        // <input type="range" id="s1" min="0" max="10" value="4" step="2">
        newSlider.type = "range";
        newSlider.min = -5;
        newSlider.max = 5;
        newSlider.value = lockSlider ? 0 : currentLatent[i];
        newSlider.step = 0.05;
        newSlider.id = "slider" + i;
        newSlider.disabled = lockSlider;
        newSlider.addEventListener('change', async function() { await changeLatentParameter(i) });

        sliderContainer.appendChild(newSlider);
    }

}

function updateDownloadLink(){
    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentLatent));
    downloadAnchorNode = document.getElementById('download-latent')
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "latent.json");
  }

//Log connection on eel server
eel.logConnection()()