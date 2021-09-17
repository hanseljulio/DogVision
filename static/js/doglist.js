let btn = document.querySelector("#btn");
let arrow = document.querySelector("#arrow");


arrow.addEventListener('click', function() {
    btn.click();
});

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();

        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});