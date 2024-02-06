// Update image on preview
document.getElementById('upload-img').addEventListener('change', function() {
    var file = this.files[0];
    if (file) {
        var reader = new FileReader();
        reader.onload = function(event) {
            image = event.target.result;
            document.getElementById('preview-img').setAttribute('src', image);
            //Upload image to python here
        }
        reader.readAsDataURL(file);
    }
});