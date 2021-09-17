let fileLabel = document.querySelector('.custom-file-label');

document.getElementById('image').onchange = function () {
    let filename = this.value.replace(/.*[\/\\]/, '');
    fileLabel.innerHTML = filename;
};

